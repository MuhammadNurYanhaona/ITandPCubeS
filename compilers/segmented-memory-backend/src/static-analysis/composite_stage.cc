#include "data_flow.h"
#include "data_access.h"
#include "sync_stat.h"
#include "task_env_stat.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/errors.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/code_constant.h"
#include "../semantics/task_space.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------ Composite Stage -------------------------------------------------------/

CompositeStage::CompositeStage(int index, Space *space, Expr *executeCond) : FlowStage(index, space, executeCond) {
	stageList = new List<FlowStage*>;
	reductionBoundary = false;	
}

void CompositeStage::addStageAtBeginning(FlowStage *stage) {
	stageList->InsertAt(stage, 0);
	stage->setParent(this);
}

void CompositeStage::addStageAtEnd(FlowStage *stage) {
	stageList->Append(stage);
	SyncStage *syncStage = dynamic_cast<SyncStage*>(stage);
	if (syncStage != NULL) {
		syncStage->setIndex(stageList->NumElements());
	}
	stage->setParent(this);
}

void CompositeStage::insertStageAt(int index, FlowStage *stage) {
	stageList->InsertAt(stage, index);
	stage->setIndex(index);
	stage->setParent(this);
}

void CompositeStage::removeStageAt(int stageIndex) { stageList->RemoveAt(stageIndex); }

bool CompositeStage::isStageListEmpty() {
	return getLastNonSyncStage() == NULL;
}

Space *CompositeStage::getLastNonSyncStagesSpace() {
	Space *space = this->space;
	FlowStage *lastNonSyncStage = getLastNonSyncStage();
	if (lastNonSyncStage != NULL) return lastNonSyncStage->getSpace();
	return space;
}

FlowStage *CompositeStage::getLastNonSyncStage() {
	int i = stageList->NumElements() - 1; 
	for (; i >= 0; i--) {
		FlowStage *stage = stageList->Nth(i);
		SyncStage *syncStage = dynamic_cast<SyncStage*>(stage);
		if (syncStage == NULL) return stage;
	}
	return NULL;

}

void CompositeStage::setStageList(List<FlowStage*> *stageList) { 
	this->stageList = stageList;
	for (int i = 0; i < stageList->NumElements(); i++) {
		stageList->Nth(i)->setParent(this);
	} 
}

void CompositeStage::addSyncStagesBeforeExecution(FlowStage *nextStage, List<FlowStage*> *stageList) {
	
	Space *previousSpace = getLastNonSyncStagesSpace();
	Space *nextSpace = nextStage->getSpace();
	
	List<Space*> *spaceTransitionChain = Space::getConnetingSpaceSequenceForSpacePair(previousSpace, nextSpace);
	if (spaceTransitionChain == NULL || spaceTransitionChain->NumElements() == 0) {
		if (!isStageListEmpty()) {
			spaceTransitionChain = new List<Space*>;
			spaceTransitionChain->Append(previousSpace);
			spaceTransitionChain->Append(nextSpace);
		}
	} 

	int nextStageIndex = nextStage->getIndex();
	if (spaceTransitionChain != NULL) {
		for (int i = 1; i < spaceTransitionChain->NumElements(); i++) {
			Space *oldSpace = spaceTransitionChain->Nth(i - 1);
			Space *newSpace = spaceTransitionChain->Nth(i);

			if (oldSpace->isParentSpace(newSpace)) {
				
				// new space is higher in the space hierarchy; so an exit from the old space should be recorded 
				// along with an entry to the new space
				SpaceEntryCheckpoint *oldCheckpoint = SpaceEntryCheckpoint::getCheckpoint(oldSpace);
				SyncStage *oldEntrySyncStage = oldCheckpoint->getEntrySyncStage();
				Hashtable<VariableAccess*> *accessLogs = getAccessLogsForSpaceInIndexLimit(oldSpace,
                        			stageList, oldCheckpoint->getStageIndex(), 
						nextStageIndex - 1, true);

				// If there is an entry sync stage for the old space then we need to populate its accessMap 
				// currectly.
				if (oldEntrySyncStage != NULL) {
					SyncStageGenerator::populateAccessMapOfEntrySyncStage(oldEntrySyncStage, accessLogs);
				}

				// if some data structures in the old space have overlapping boundary regions among their parts
				// and some of those data structures have been modified, a ghost regions sync is needed that 
				// operate on the old space as overlapping boundaries should be synchronized at each space exit
				SyncStage *reappearanceSync = 
						SyncStageGenerator::generateReappearanceSyncStage(oldSpace, accessLogs);
				if (reappearanceSync != NULL) {
					addStageAtEnd(reappearanceSync);
				}
				
				// generate and add to the list all possible sync stages that are required due to the exit from 
				// the old space
				SpaceEntryCheckpoint::removeACheckpoint(oldSpace);
				List<SyncStage*> *exitSyncs = SyncStageGenerator::generateExitSyncStages(oldSpace, accessLogs);
				for (int i = 0; i < exitSyncs->NumElements(); i++) {
					SyncStage *exitSyncStage = exitSyncs->Nth(i);
					addStageAtEnd(exitSyncStage);
				}
		
				// generate and add any potential return sync stage to the new space
				accessLogs = getAccessLogsForReturnToSpace(newSpace, stageList, nextStageIndex - 1);
				SyncStage *returnSync = SyncStageGenerator::generateReturnSyncStage(newSpace, accessLogs);
				if (returnSync != NULL) {
					addStageAtEnd(returnSync);
				}
			} else if (newSpace->isParentSpace(oldSpace)) {
				// Old space is higher in the space hierarchy; so an entry to the new space should be recorded. 
				// The entry sync stage here, if present, is just a place holder. Later on during exit its 
				// accessLog is filled with appropriate data
				SyncStage *entrySyncStage = SyncStageGenerator::generateEntrySyncStage(newSpace);
				SpaceEntryCheckpoint *checkpoint = 
						SpaceEntryCheckpoint::addACheckpointIfApplicable(newSpace, nextStageIndex);
				checkpoint->setEntrySyncStage(entrySyncStage);
				if (entrySyncStage != NULL) {
					addStageAtEnd(entrySyncStage);
				}
			} else if (oldSpace != newSpace) {
				std::cout << "We should never have a disjoint space transition chain\n";
				std::cout << "Old Space " << oldSpace->getName() << '\n';
				std::cout << "New Space " << newSpace->getName() << '\n';
				std::exit(EXIT_FAILURE);
			}
		}
	}
}

void CompositeStage::addSyncStagesOnReturn(List<FlowStage*> *stageList) {
        
	Space *previousSpace = getLastNonSyncStagesSpace();
	Space *currentSpace = getSpace();
	List<Space*> *spaceTransitionChain = Space::getConnetingSpaceSequenceForSpacePair(previousSpace, currentSpace);
	if (spaceTransitionChain == NULL || spaceTransitionChain->NumElements() == 0) return;
	
	int lastStageIndex = getLastNonSyncStage()->getIndex();
	for (int i = 1; i < spaceTransitionChain->NumElements(); i++) {
        	
		Space *oldSpace = spaceTransitionChain->Nth(i - 1);
        	Space *newSpace = spaceTransitionChain->Nth(i);

		SpaceEntryCheckpoint *oldCheckpoint = SpaceEntryCheckpoint::getCheckpoint(oldSpace);
                SyncStage *oldEntrySyncStage = oldCheckpoint->getEntrySyncStage();
                Hashtable<VariableAccess*> *accessLogs = getAccessLogsForSpaceInIndexLimit(oldSpace,
                		stageList, oldCheckpoint->getStageIndex(), lastStageIndex, true);

                // If there is an entry sync stage for the old space then we need to populate its accessMap currectly.
                if (oldEntrySyncStage != NULL) {
                	SyncStageGenerator::populateAccessMapOfEntrySyncStage(oldEntrySyncStage, accessLogs);
                }

		// if some data structures in the old space have overlapping boundary regions among their parts and some of 
		// those data structures have been modified, a ghost regions sync is needed that operate on the old space as 
		// overlapping boundaries should be synchronized at each space exit
		SyncStage *reappearanceSync = 
				SyncStageGenerator::generateReappearanceSyncStage(oldSpace, accessLogs);
		if (reappearanceSync != NULL) {
			addStageAtEnd(reappearanceSync);
		}

                // generate and add to the list all possible sync stages that are required due to the exit from the old space
                SpaceEntryCheckpoint::removeACheckpoint(oldSpace);
                List<SyncStage*> *exitSyncs = SyncStageGenerator::generateExitSyncStages(oldSpace, accessLogs);
                for (int i = 0; i < exitSyncs->NumElements(); i++) {
                	SyncStage *exitSyncStage = exitSyncs->Nth(i);
                	addStageAtEnd(exitSyncStage);
                }

                // generate and add any potential return sync stage to the new space
                accessLogs = getAccessLogsForReturnToSpace(newSpace, stageList, lastStageIndex);
                SyncStage *returnSync = SyncStageGenerator::generateReturnSyncStage(newSpace, accessLogs);
                if (returnSync != NULL) {
                	addStageAtEnd(returnSync);
                }	
	}	
}

void CompositeStage::print(int indent) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("Flow Stage: %s (Space: %s)", name, space->getName());
	if (executeCond != NULL) printf(" Conditionally Executed");
	printf("\n");
	for (int i = 0; i < stageList->NumElements(); i++) {
		stageList->Nth(i)->print(indent + 1);
	}
	dataDependencies->print(indent + 1);
}

void CompositeStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		stageList->Nth(i)->performDependencyAnalysis(hierarchy);
	}
}

void CompositeStage::performEpochUsageAnalysis() {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->performEpochUsageAnalysis();
		List<const char*> *stageEpochList = stage->getEpochDependentVarList();
		string_utils::combineLists(epochDependentVarList, stageEpochList);
	}
}

void CompositeStage::populateReductionMetadata(PartitionHierarchy *lpsHierarchy) {
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->populateReductionMetadata(lpsHierarchy);
	}
}

void CompositeStage::extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos) {
	
	// composite stages should not add their own lists of reduction metadata into the argument list as their
	// own lists have been created by extracting metadata from nested execution stages.
	
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->extractAllReductionInfo(reductionInfos);
	}
}

void CompositeStage::reorganizeDynamicStages() {
	
	int currentStageIndex = 0;
	while (currentStageIndex < stageList->NumElements()) {
		
		FlowStage *stage = stageList->Nth(currentStageIndex);
		
		// do the reorganization for nested stages wherever necessary
		stage->reorganizeDynamicStages();

		stage->setIndex(currentStageIndex);
		currentStageIndex++;

		if (!stage->getSpace()->isDynamic()) continue;
		SyncStage *syncStage = dynamic_cast<SyncStage*>(stage);
		if (syncStage == NULL || !syncStage->isLoaderSync()) continue;
		
		// gather the stages that should be included in the generated composite stage
		List<Expr*> *activationConditions = new List<Expr*>;
		List<FlowStage*> *toBeNestedStages = new List<FlowStage*>;
		int stageListStart = currentStageIndex;
		Space *dynamicSpace = stage->getSpace();
		while (stageList->NumElements() > stageListStart) {
			FlowStage *newStage = stageList->Nth(stageListStart);
			Space *newSpace = newStage->getSpace();
			if (newSpace != dynamicSpace && !newSpace->isParentSpace(dynamicSpace)) break;
			if (newSpace == dynamicSpace && newStage->getExecuteCondition() != NULL) {
				activationConditions->Append(newStage->getExecuteCondition());
			}
			toBeNestedStages->Append(newStage);
			// do the reorganization for nested stages wherever necessary
			newStage->reorganizeDynamicStages();
			// remove the stage from current list
			removeStageAt(stageListStart);
		}
		
		// generate an updated execute condition for the new to be added composite stage
		Expr *overallCondition = NULL;
		if (activationConditions->NumElements() == 1) {
			for (int i = 0; i < toBeNestedStages->NumElements(); i++) {
				FlowStage *newStage = toBeNestedStages->Nth(i);
				if (newStage->getSpace() == dynamicSpace) {
					// TODO probably the correct solution is to wrap up the computation
					// inside within an conditional block that test the validity of 
					// entering into this stage. That way we will avoid the problem of
					// reevaluating active LPUs for this stage and at the same time ensure
					// that we are not going to execute any unintended code.	
					newStage->setExecuteCondition(NULL);
				}
			}
			overallCondition = activationConditions->Nth(0);
		} else if (activationConditions->NumElements() > 1) {
			overallCondition = activationConditions->Nth(0);
			for (int i = 1; i < activationConditions->NumElements(); i++) {
				Expr *nextCondition = activationConditions->Nth(i);
				overallCondition = new LogicalExpr(overallCondition, OR, 
						nextCondition, *overallCondition->GetLocation());
			}	
		}

		// generate the composite stage
		CompositeStage *generatedStage = new CompositeStage(currentStageIndex - 1, 
				dynamicSpace, overallCondition);
		generatedStage->setName("\"Dynamic Computation\"");
		generatedStage->setStageList(toBeNestedStages);
		generatedStage->addStageAtBeginning(syncStage);
		
		// remove the sync stage from current list and add the composite stage in its place
		removeStageAt(currentStageIndex - 1);
		insertStageAt(currentStageIndex - 1, generatedStage);
	}
}

void CompositeStage::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->fillInTaskEnvAccessList(envAccessList);
	}
}

void CompositeStage::prepareTaskEnvStat(TaskEnvStat *taskStat, Hashtable<VariableAccess*> *accessMap) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->prepareTaskEnvStat(taskStat);
	}
}

List<List<FlowStage*>*> *CompositeStage::getConsecutiveNonLPSCrossingStages() {
	
	List<List<FlowStage*>*> *stageGroups = new List<List<FlowStage*>*>;
	Space *currentSpace = stageList->Nth(0)->getSpace();
	List<FlowStage*> *currentGroup = new List<FlowStage*>;
	currentGroup->Append(stageList->Nth(0));

	for (int i = 1; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		RepeatCycle *repeatCycle = dynamic_cast<RepeatCycle*>(stage);
		// repeat cycles are never put into any compiler generated group to keep the semantics of the
		// program intact
		if (repeatCycle != NULL) {
			// add the so far group into the stage groups list if it is not empty
			if (currentGroup->NumElements() > 0) stageGroups->Append(currentGroup);
			// create a new, isolated group for repeat and add that in the list too 
			List<FlowStage*> *repeatGroup = new List<FlowStage*>;
			repeatGroup->Append(repeatCycle);
			stageGroups->Append(repeatGroup);
			// then reset the current group
			currentGroup = new List<FlowStage*>;
		// if the stage is executing in a different LPS then create a new group
		}	
		else if (stage->getSpace() != currentSpace) {
			currentSpace = stage->getSpace();
			if (currentGroup->NumElements() > 0) stageGroups->Append(currentGroup);
			currentGroup = new List<FlowStage*>;
			currentGroup->Append(stage);
		// otherwise add the stage in the current group if there is no synchronization dependency from
		// current stage to any stage already within the group. If there is a dependency then create a
		// new group. Note that this policy helps us to drag down LPU-LPU synchronization dependencies 
		// as transition has been made between computation stages into PPU-PPU dependencies. 
		} else {
			bool dependencyExists = false;
			for (int j = 0; j < currentGroup->NumElements(); j++) {
				FlowStage *earlierStage = currentGroup->Nth(j);
				if (earlierStage->isDependentStage(stage) 
						|| stage->isDependentStage(earlierStage)) {
					dependencyExists = true;
					break;
				}
			}
			if (dependencyExists) {
				stageGroups->Append(currentGroup);
				currentGroup = new List<FlowStage*>;
				currentGroup->Append(stage);
			} else 	currentGroup->Append(stage);
		}
	}
	if (currentGroup->NumElements() > 0) stageGroups->Append(currentGroup);
	return stageGroups;
}

void CompositeStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	int nextIndentation = indentation;
	std::ostringstream nextIndent;
	nextIndent << indent.str();

	// if the index is 0 then it is the first composite stage representing the entire compution. We declare
	// any synchronization counter that is applicable outside all repeat-cycle boundaries
	if (this->index == 0) {
		declareSynchronizationCounters(stream, indentation, this->repeatIndex + 1);
	}

	// if their is an LPS transition due to entering this stage then create a while loop traversing LPUs
	// of newly entered LPS
	if (this->space != containerSpace) {
		const char *spaceName = space->getName();
		nextIndentation++;
		nextIndent << '\t';
		// create a new local scope for traversing LPUs of this new scope
		stream << std::endl;
		stream << indent.str() << "{ // scope entrance for iterating LPUs of Space ";
		stream << spaceName << "\n";
		// declare a new variable for tracking the last LPU id
		stream << indent.str() << "int space" << spaceName << "LpuId = INVALID_ID" << stmtSeparator;
		// declare another variable to track the iteration number of the while loop
		stream << indent.str() << "int space" << spaceName << "Iteration = 0" << stmtSeparator;
		// declare a new variable to hold on to current LPU of this LPS
		stream << indent.str() << "Space" << spaceName << "_LPU *space" << spaceName << "Lpu = NULL";
		stream << stmtSeparator;
		// declare another variable to assign the value of get-Next-LPU call
		stream << indent.str() << "LPU *lpu = NULL" << stmtSeparator;
		stream << indent.str() << "while((lpu = threadState->getNextLpu(";
		stream << "Space_" << spaceName << paramSeparator << "Space_" << containerSpace->getName();
		stream << paramSeparator << "space" << spaceName << "LpuId)) != NULL) {\n";
		// cast the common LPU variable to LPS specific LPU		
		stream << nextIndent.str() << "space" << spaceName << "Lpu = (Space" << spaceName;
		stream  << "_LPU*) lpu" << stmtSeparator;
	}

	// Check if there is any activating condition attached with this container stage. If there is such a
	// condition then the stages inside should be executed only if the activating condition evaluates to true.
	if (executeCond != NULL) {
		
		// first get a hold of the LPU reference
		std::ostringstream lpuVarName;
		lpuVarName << "space" << space->getName() << "Lpu->";
		
		// then generate local variables for any array been accessed within the condition
		List<FieldAccess*> *fields = executeCond->getTerminalFieldAccesses();
		for (int i = 0; i < fields->NumElements(); i++) {
			DataStructure *structure = space->getLocalStructure(fields->Nth(i)->getField()->getName());
			if (structure == NULL) continue;
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			if (array == NULL) continue;
			const char *arrayName = array->getName();
        		int dimensions = array->getDimensionality();
        		stream << nextIndent.str() << "Dimension ";
                	stream  << arrayName << "PartDims[" << dimensions << "];\n";
        		stream << nextIndent.str() << "Dimension ";
                	stream  << arrayName << "StoreDims[" << dimensions << "];\n";
                	for (int j = 0; j < dimensions; j++) {
                		stream << indent.str();
               			stream << arrayName << "PartDims[" << j << "] = " << lpuVarName.str();
                        	stream << arrayName << "PartDims[" << j << "].partition;\n";
                		stream << indent.str();
               			stream << arrayName << "StoreDims[" << j << "] = " << lpuVarName.str();
                        	stream << arrayName << "PartDims[" << j << "].storage;\n";
        		}
		}
		
		// now set the name transformer's LPU prefix properly so that if the activate condition involves
		// accessing elements of the LPU, it works correctly
		ntransform::NameTransformer::transformer->setLpuPrefix(lpuVarName.str().c_str());
	
		// then generate an if condition for condition checking
		stream << nextIndent.str() << "if(!(";
		std::ostringstream conditionStream;
		executeCond->translate(conditionStream, nextIndentation, 0, space);
		stream << conditionStream.str();
		stream << ")) {\n";
		// we skip the current LPU if the condition evaluates to false	
		stream << nextIndent.str() << "\tcontinue" << stmtSeparator;
		stream << nextIndent.str() << "}\n";	
	}

	// if the current composite stage is boundary for some reduction operations then initialize partial results
	// of reductions for the executing PPU controller thread
	if (reductionBoundary) {
		initializeReductionResults(stream, nextIndentation);
	}
	
	// Iterate over groups of flow stages where each group executes within a single LPS. This scheme has the
	// consequence of generating LPU only one time for all stages of a group then execute all of them before
	// proceed to the next LPU 
	List<List<FlowStage*>*> *stageGroups = getConsecutiveNonLPSCrossingStages();
	
	for (int i = 0; i < stageGroups->NumElements(); i++) {
		
		List<FlowStage*> *currentGroup = stageGroups->Nth(i);

		// retrieve all data dependencies, and sort them to ensure waitings for updates happen in order
		List<SyncRequirement*> *dataDependencies = getDataDependeciesOfGroup(currentGroup);
		dataDependencies = SyncRequirement::sortList(dataDependencies);
		
		// separate dependencies into communication and synchronization dependencies then take appropriate 
		// actions based on type (the simplified implementation does nothing with the synchronization 
		// dependencies)
		int segmentedPPS = space->getSegmentedPPS();
		List<SyncRequirement*> *commDependencies = new List<SyncRequirement*>;
		List<SyncRequirement*> *syncDependencies = new List<SyncRequirement*>;
		SyncRequirement::separateCommunicationFromSynchronizations(segmentedPPS, 
				dataDependencies, commDependencies, syncDependencies);
		generateDataReceivesForGroup(stream, nextIndentation, commDependencies);
		
		// retrieve all shared data update signals that need to be activated if stages in the group execute
		List<SyncRequirement*> *updateSignals = getUpdateSignalsOfGroup(currentGroup);
		// mark these signals as signaled so that they are not reactivated within the nested code
		for (int j = 0; j < updateSignals->NumElements(); j++) {
			updateSignals->Nth(j)->signal();
		}
		// sort the update signals to ensure waiting for signal clearance (equivalent to get signals from the
		// readers that all of them have finished reading the last update) happens in proper order
		updateSignals = SyncRequirement::sortList(updateSignals);

		// divide the signals between those issuing communications and those that do not  
		List<SyncRequirement*> *commSignals = new List<SyncRequirement*>;
		List<SyncRequirement*> *syncSignals = new List<SyncRequirement*>;
		SyncRequirement::separateCommunicationFromSynchronizations(segmentedPPS, 
				updateSignals, commSignals, syncSignals);

		// simplified implementation	
		//-------------------------------------------------------------------------------------------------
		// If there is any reactivating condition that need be checked before we let the flow of control 
		// enter the nested stages then we wait for those condition clearance.
		genSimplifiedWaitingForReactivationCode(stream, nextIndentation, updateSignals);
		//-------------------------------------------------------------------------------------------------

		// Sync stages -- not synchronization dependencies -- that dictate additional data movement 
		// operations are not needed (according to our current (date: Jan-30-2015) understanding); so we
		// filter them out.	
		currentGroup = filterOutSyncStages(currentGroup);
		if (currentGroup->NumElements() == 0) {
			// before rulling sync stages out, we need to ensure that whatever signals they were supposed
			// issue are by-default issued and whatever data they were supposed to send are sent
			generateSignalCodeForGroupTransitions(stream, nextIndentation, syncSignals);
			generateDataSendsForGroup(stream, nextIndentation, commSignals); 
			continue;
		}

		// there should be a special checking for repeat loops. Repeat loops will have the while loop that
		// iterates over LPUs of the LPS under concern inside its body if it does not use LPS dependent 
		// variables in repeat evaluation process. Otherwise, it should follow the normal code generation
		// procedure followed for other stages
		bool groupIsLpsIndRepeat = false; 
		if (currentGroup->NumElements() == 1) {
			FlowStage *stage = currentGroup->Nth(0);
			RepeatCycle *repeatCycle = dynamic_cast<RepeatCycle*>(stage);
			if (repeatCycle != NULL && !repeatCycle->isLpsDependent()) {
				Space *repeatSpace = repeatCycle->getSpace();
				// temporarily the repeat cycle is assigned to be executed in current LPS to ensure
				// that LPU iterations for stages inside it proceed accurately
				repeatCycle->changeSpace(this->space);
				repeatCycle->generateInvocationCode(stream, nextIndentation, space);
				// reset the repeat cycle's LPS to the previous one once code generation is done
				repeatCycle->changeSpace(repeatSpace);
				groupIsLpsIndRepeat = true;
			}	
		}

		if (!groupIsLpsIndRepeat) {
			Space *groupSpace = currentGroup->Nth(0)->getSpace();
			// if the LPS of the group is not the same of this one then we need to consider LPS entry, 
			// i.e., include a while loop for the group to traverse over the LPUs
			if (groupSpace != space) {
				// create a local composite stage to apply the logic of this function recursively
				CompositeStage *tempStage = new CompositeStage(-1, groupSpace, NULL);
				tempStage->setStageList(currentGroup);
				tempStage->generateInvocationCode(stream, nextIndentation, space);
				delete tempStage;
			// otherwise the current LPU of this composite stage will suffice and we execute all the 
			// nested stages one after one.	 
			} else {
				for (int j = 0; j < currentGroup->NumElements(); j++) {
					FlowStage *stage = currentGroup->Nth(j);
					stage->generateInvocationCode(stream, nextIndentation, space);
				}
			}
		}

		// simplified implementation	
		//-------------------------------------------------------------------------------------------------
		// generate code for signaling updates and waiting on those updates for any shared variable change
		// made by stages within current group  
		genSimplifiedSignalsForGroupTransitionsCode(stream, nextIndentation, syncSignals);
		//-------------------------------------------------------------------------------------------------
		
		// communicate any update of shared data	
		generateDataSendsForGroup(stream, nextIndentation, commSignals); 

		// commented out this code as the barrier based priliminary implementation does not need this
		/*-------------------------------------------------------------------------------------------------
		// finally if some updater stages can be executed again because current group has finished execution
		// then we need to enable the reactivation signals
		generateCodeForReactivatingDataModifiers(stream, nextIndentation, syncDependencies);
		//-----------------------------------------------------------------------------------------------*/
	}
	
	// if the current composite stage is boundary for some reduction operations then execute the final step of
	// reduction
	if (reductionBoundary) {
		executeFinalStepOfReductions(stream, nextIndentation);
	}

	// close the while loop if applicable
	if (space != containerSpace) {
		const char *spaceName = space->getName();
		// update the iteration number and next LPU id
		stream << nextIndent.str() << "space" << spaceName << "LpuId = space" << spaceName;
		stream << "Lpu->id" << stmtSeparator;	
		stream << nextIndent.str() << "space" << spaceName << "Iteration++" << stmtSeparator;
		stream << indent.str() << "}\n";
		// at the end remove checkpoint if the container LPS is not the root LPS
		if (!containerSpace->isRoot()) {
			stream << indent.str() << "threadState->removeIterationBound(Space_";
			stream << containerSpace->getName() << ')' << stmtSeparator;
		}
		// exit from the scope
		stream << indent.str() << "} // scope exit for iterating LPUs of Space ";
		stream << space->getName() << "\n";
	}	
}

// A composite stage is a group entry if it has flow stages of multiple LPSes inside or any stage inside it
// is a group entry.
bool CompositeStage::isGroupEntry() {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		if (stage->isGroupEntry()) return true;
	}
	List<List<FlowStage*>*> *stageGroups = getConsecutiveNonLPSCrossingStages();
	return (stageGroups->NumElements() > 1 || space != stageGroups->Nth(0)->Nth(0)->getSpace());
}

void CompositeStage::setLpsExecutionFlags() {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->setLpsExecutionFlags();
	}
}

void CompositeStage::calculateLPSUsageStatistics() {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->calculateLPSUsageStatistics();
	}	
}

void CompositeStage::analyzeSynchronizationNeeds() {	
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->analyzeSynchronizationNeeds();
	}
}

void CompositeStage::retriveExternCodeBlocksConfigs(IncludesAndLinksMap *externConfigMap) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->retriveExternCodeBlocksConfigs(externConfigMap);
	}	
}

int CompositeStage::getHighestNestedStageIndex() {
	int stageCount = stageList->NumElements();
	FlowStage *lastStage = stageList->Nth(stageCount - 1);
	CompositeStage *nestedCompositeStage = dynamic_cast<CompositeStage*>(lastStage);
	if (nestedCompositeStage == NULL) {
		return lastStage->getIndex();
	} else {
		return nestedCompositeStage->getHighestNestedStageIndex();
	}
}

void CompositeStage::analyzeSynchronizationNeedsForComposites() {	
	
	// evaluate synchronization requirements and dependencies within its nested composite stages
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
		if (compositeStage != NULL) {
			compositeStage->analyzeSynchronizationNeedsForComposites();
		}
	}
	
	// get the first and last flow index within the group to know the boundary of this composite stage
	int beginIndex = stageList->Nth(0)->getIndex();
	int endIndex = getHighestNestedStageIndex();
	
	// start extracting boundary crossing synchronizations out
	synchronizationReqs = new StageSyncReqs(this);

	// iterate over the nested stages and extract any boundary crossing synchronization founds within and assign 
	// that to current composite stage
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		StageSyncReqs *nestedSyncs = stage->getAllSyncRequirements();
		List<VariableSyncReqs*> *varSyncList = nestedSyncs->getVarSyncList();
		for (int j = 0; j < varSyncList->NumElements(); j++) {
			VariableSyncReqs *varSyncs = varSyncList->Nth(j);
			List<SyncRequirement*> *syncReqList = varSyncs->getSyncList();
			for (int k =  0; k < syncReqList->NumElements(); k++) {
				SyncRequirement *syncReq = syncReqList->Nth(k);
				FlowStage *waitingStage = syncReq->getDependencyArc()->getSignalSink();
				int sinkIndex = waitingStage->getIndex();
				if (sinkIndex < beginIndex || sinkIndex > endIndex) {
					synchronizationReqs->addVariableSyncReq(varSyncs->getVarName(), 
							syncReq, false);
					// update the nesting index of the arc so that it can be discovered later by 
					// its nesting index
					syncReq->getDependencyArc()->setNestingIndex(this->repeatIndex);
					// then update the signal source for the arc to indicate inside that it has 
					// been lifted up
					syncReq->getDependencyArc()->setSignalSrc(this);
				}
			}
		}
	}
}

void CompositeStage::setupReductionBoundaryStages(PartitionHierarchy *lpsHierarchy) {
	
	int nestedStageCount = stageList->NumElements();
	for (int i = 0; i < nestedStageCount; i++) {
		FlowStage *stage = stageList->Nth(i);
		
		// if the nested stage is a composite stage then recursively invoke the function inside that to expand
		// the subflow it contains
		CompositeStage *subflow = dynamic_cast<CompositeStage*>(stage);
		if (subflow != NULL) {
			subflow->setupReductionBoundaryStages(lpsHierarchy);
			continue;
		}

		// if the nested stage is not an execution stage or does not have any reduction then no modification
		// is needed
		ExecutionStage *computation = dynamic_cast<ExecutionStage*>(stage);
		if (computation == NULL || !computation->hasNestedReductions()) {
			continue;
		}

		// find the topmost LPS where the root of some reduction range is located
		List<ReductionMetadata*> *reductions = computation->getNestedReductions();
		Space *topmostReductionLps = computation->getSpace();
		for (int j = 0; j < reductions->NumElements(); j++) {
			
			ReductionMetadata *metadata = reductions->Nth(j);
			Space *candidateLps = metadata->getReductionRootLps();
			
			// make sure that the reduction operation's root LPS has not been incorrectly placed to some LPS
			// above the LPS the subflow represented by the current composite stage is supposed to execute
			Space *commonAncestor = lpsHierarchy->getCommonAncestor(candidateLps, topmostReductionLps);
			if (commonAncestor != this->space && !commonAncestor->isParentSpace(this->space)) {
				ReportError::ReductionRangeInvalid(metadata->getLocation(), 
						candidateLps->getName(), space->getName(), false);
			}

			if (topmostReductionLps->isParentSpace(candidateLps)) {
				topmostReductionLps = candidateLps;
			}
		}

		// setup a new reduction boundary composite stage on the topmost reduction root LPS and move the compu-
		// tation inside it
		CompositeStage *reductionBoundary = new CompositeStage(-1, topmostReductionLps, NULL);
		reductionBoundary->setName("Reduction Boundary");
		reductionBoundary->flagAsReductionBoundary();
		this->removeStageAt(i);
		reductionBoundary->addStageAtBeginning(computation);
		this->insertStageAt(i, reductionBoundary);

		// move reduction annotations having root at topmostReductionLps from the computation to the reduction 
		// boundary stage as the letter will do variable setup and final reduction.
		List<ReductionMetadata*> *boundaryList = reductionBoundary->getNestedReductions();
		int j = 0;
		while (j < reductions->NumElements()) {
			ReductionMetadata *metadata = reductions->Nth(j);
			Space *reductionRootLps = metadata->getReductionRootLps();
			if (reductionRootLps == topmostReductionLps) {
				boundaryList->Append(metadata);
				reductions->RemoveAt(j);
			} else j++;
		}
		
		// recursively call the function in the boundary stage to expand reductions with lower root LPSes.
		reductionBoundary->setupReductionBoundaryStages(lpsHierarchy);		
	}
}

void CompositeStage::deriveSynchronizationDependencies() {
	
	// derive dependencies within its nested composite stages
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
		if (compositeStage != NULL) {
			compositeStage->deriveSynchronizationDependencies();
		}
	}

	// get the first and last flow index within the group to know the boundary of this composite stage
	int beginIndex = stageList->Nth(0)->getIndex();
	int endIndex = getHighestNestedStageIndex();
	
	// then check those stages and draw out any synchronization dependencies they have due to changes made in
	// stages outside this composite stage as the dependencies of the current composite stage itself.
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		StageSyncDependencies *nestedDependencies = stage->getAllSyncDependencies();
		List<SyncRequirement*> *dependencyList = nestedDependencies->getDependencyList();
		for (int j = 0; j < dependencyList->NumElements(); j++) {
			SyncRequirement *sync = dependencyList->Nth(j);
			FlowStage *syncSource = sync->getDependencyArc()->getSignalSrc();
			// if the source's index is within the nesting boundary then the dependency is internal
			// otherwise, it is external and we have to pull it out.
			int syncIndex = syncSource->getIndex();
			if (syncIndex < beginIndex || syncIndex > endIndex) {
				this->syncDependencies->addDependency(sync);
				// a pulled out dependency's arc should be updated to reflect that the current
				// composite stage is its sink
				sync->getDependencyArc()->setSignalSink(this);
			}
		}
	}
}

void CompositeStage::printSyncRequirements(int indentLevel) {
	for (int i = 0; i < indentLevel; i++) std::cout << '\t';
	std::cout << "Stage: " << name << ":\n";
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->printSyncRequirements(indentLevel + 1);
	}	
}

int CompositeStage::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {
	int nextIndex = FlowStage::assignIndexAndGroupNo(currentIndex, currentGroupNo, currentRepeatCycle);
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		nextIndex = stage->assignIndexAndGroupNo(nextIndex, this->index, currentRepeatCycle);
	}
	return nextIndex;	
}

List<FlowStage*> *CompositeStage::filterOutSyncStages(List<FlowStage*> *originalList) {
	List<FlowStage*> *filteredList = new List<FlowStage*>;
	for (int i = 0; i < originalList->NumElements(); i++) {
		FlowStage *stage = originalList->Nth(i);
		SyncStage *sync = dynamic_cast<SyncStage*>(stage);
		if (sync == NULL) filteredList->Append(stage);
	}
	return filteredList;
}

List<SyncRequirement*> *CompositeStage::getDataDependeciesOfGroup(List<FlowStage*> *group) {
	List<SyncRequirement*> *syncList = new List<SyncRequirement*>;
	for (int i = 0; i < group->NumElements(); i++) {
		FlowStage *stage = group->Nth(i);
		StageSyncDependencies *sync = stage->getAllSyncDependencies();
		List<SyncRequirement*> *activeDependencies = sync->getActiveDependencies();
		syncList->AppendAll(activeDependencies);
	}
	return syncList;
}

void CompositeStage::generateDataReceivesForGroup(std::ofstream &stream, int indentation,
                        List<SyncRequirement*> *commDependencies) {
	
	// Note that all synchronization we are dealing here is strictly within the boundary of composite stage under
	// concern. Now if there is a synchronization dependency to a nested stage for the execution of another stage
	// that comes after it, it means that the dependency is between a latter iteration on the dependent stage on
	// an earlier iteration on the source stage. This is because, otherwise the dependent stage will execute before
	// the source stage and there should not be any dependency. Now, such dependency is only valid for iterations 
	// except the first one. So there should be a checking on iteration number before we apply waiting on such
	// dependencies. On the other hand, if the source stage executes earlier than the dependent stage then it applies
	// always, i.e., it is independent of any loop iteration, if exists, that sorrounds the composite stage.
	// Therefore, we use following two lists to partition the list of communication requirements into forward and
	// backward dependencies. 
	List<SyncRequirement*> *forwardDependencies = new List<SyncRequirement*>;
	List<SyncRequirement*> *backwardDependencies = new List<SyncRequirement*>;

	for (int i = 0; i < commDependencies->NumElements(); i++) {
		SyncRequirement *comm = commDependencies->Nth(i);
		if (!comm->isActive()) continue;
		DependencyArc *arc = comm->getDependencyArc();
		FlowStage *source = arc->getSource();
		FlowStage *destination = arc->getDestination();
		if (source->getIndex() < destination->getIndex()) {
			forwardDependencies->Append(comm);
		} else backwardDependencies->Append(comm);
	}

	std::ostringstream indentStr;
	for (int i = 0; i < indentation; i++) indentStr << '\t';
	
	if (commDependencies->NumElements() > 0) {
		stream << std::endl << indentStr.str() << "// waiting on data reception\n";
	}
	
	// Write the code for forwards communication requirements.
	for (int i = 0; i < forwardDependencies->NumElements(); i++) {
		
		SyncRequirement *comm = forwardDependencies->Nth(i);

		// if the data send for this communicator has been replaced with some earlier communicator then the data receive
		// is done using that earlier communicator too 
		SyncRequirement *commReplacement = comm->getReplacementSync();
		bool signalReplaced = false;
		if (commReplacement != NULL) {
			comm = commReplacement;
			signalReplaced = true;
		}	

		int commIndex = comm->getIndex();
		Space *dependentLps = comm->getDependentLps();
		stream << indentStr.str() << "if (threadState->isValidPpu(Space_" << dependentLps->getName();
		stream << ")) {\n";
		stream << indentStr.str() << indent << "Communicator *communicator = threadState->getCommunicator(\"";
		stream << comm->getDependencyArc()->getArcName() << "\")" << stmtSeparator;
		stream << indentStr.str() << indent << "if (communicator != NULL) {\n";
		stream << indentStr.str() << doubleIndent << "communicator->receive(REQUESTING_COMMUNICATION";
		stream << paramSeparator << "commCounter" << commIndex << ")" << stmtSeparator; 
		stream << indentStr.str() << indent << "}\n";
		stream << indentStr.str() << "}\n";

		// The counter should be advanced regardless of this PPU's participation in communication to keep the counter
		// value uniform across all PPUs and segments. The exeception is when the data send signal has been replaced
		// by some other communicator. This is because, we will have two receive calls for the replacement communicator
		// then and we would want to make the second call to bypass any data processing. 
		if (!signalReplaced) {
			stream << indentStr.str() << "commCounter" << commIndex << "++" << stmtSeparator;
		}
	}

	// write the code for backword sync requirements within an if block
	if (backwardDependencies->NumElements() > 0) {
		stream << indentStr.str() << "if (repeatIteration > 0) {\n";
		for (int i = 0; i < backwardDependencies->NumElements(); i++) {
			
			SyncRequirement *comm = backwardDependencies->Nth(i);
			
			SyncRequirement *commReplacement = comm->getReplacementSync();
			bool signalReplaced = false;
			if (commReplacement != NULL) {
				comm = commReplacement;
				signalReplaced = true;
			}	

			int commIndex = comm->getIndex();
			Space *dependentLps = comm->getDependentLps();
			stream << indentStr.str() << indent; 
			stream << "if (threadState->isValidPpu(Space_" << dependentLps->getName();
			stream << ")) {\n";
			stream << indentStr.str() << doubleIndent;
			stream << "Communicator *communicator = threadState->getCommunicator(\"";
			stream << comm->getDependencyArc()->getArcName() << "\")" << stmtSeparator;
			stream << indentStr.str() << doubleIndent << "if (communicator != NULL) {\n";
			stream << indentStr.str() << tripleIndent;
			stream << "communicator->receive(REQUESTING_COMMUNICATION";
			stream << paramSeparator << "commCounter" << commIndex << ")" << stmtSeparator; 
			stream << indentStr.str() << doubleIndent << "}\n";
			stream << indentStr.str() << indent << "}\n";

			// again the counter is advanced outside the if condition to keep it in sync with all other PPUs
			if (!signalReplaced) {
				stream << indentStr.str() << indent << "commCounter";
				stream << commIndex << "++" << stmtSeparator;
			}
		}
		stream << indentStr.str() << "}\n";
	}

	// finally deactive all sync dependencies as they are already been taken care of here	
	for (int i = 0; i < commDependencies->NumElements(); i++) {
		SyncRequirement *comm = commDependencies->Nth(i);
		comm->deactivate();
	}
}

void CompositeStage::generateSyncCodeForGroupTransitions(std::ofstream &stream, int indentation,
                        List<SyncRequirement*> *syncDependencies) {

	// Note that all synchronization we are dealing here is strictly within the boundary of composite stage under
	// concern. Now if there is a synchronization dependency to a nested stage for the execution of another stage
	// that comes after it, it means that the dependency is between a latter iteration on the dependent stage on
	// an earlier iteration on the source stage. This is because, otherwise the dependent stage will execute before
	// the source stage and there should not be any dependency. Now, such dependency is only valid for iterations 
	// except the first one. So there should be a checking on iteration number before we apply waiting on such
	// dependencies. On the other hand, if the source stage executes earlier than the dependent stage then it applies
	// always, i.e., it is independent of any loop iteration, if exists, that sorrounds the composite stage.
	// Therefore, we use following two lists to partition the list of synchronization requirements into forward and
	// backward dependencies. 
	List<SyncRequirement*> *forwardSyncs = new List<SyncRequirement*>;
	List<SyncRequirement*> *backwardSyncs = new List<SyncRequirement*>;

	for (int i = 0; i < syncDependencies->NumElements(); i++) {
		SyncRequirement *sync = syncDependencies->Nth(i);
		if (!sync->isActive()) continue;
		DependencyArc *arc = sync->getDependencyArc();
		FlowStage *source = arc->getSource();
		FlowStage *destination = arc->getDestination();
		if (source->getIndex() < destination->getIndex()) {
			forwardSyncs->Append(sync);
		} else backwardSyncs->Append(sync);
	}

	std::ostringstream indentStr;
	for (int i = 0; i < indentation; i++) indentStr << '\t';
	
	if (syncDependencies->NumElements() > 0) {
		stream << std::endl << indentStr.str() << "// waiting for synchronization signals\n";
	}
	
	// Write the code for forwards sync requirements. Now we are just printing the requirements that should be replaced
	// with actual waiting logic later. Note the if condition that ensures that the thread is supposed to wait for the 
	// signal by checking that if it has a valid ppu-id corresponding to the LPS under concern.
	for (int i = 0; i < forwardSyncs->NumElements(); i++) {
		SyncRequirement *sync = forwardSyncs->Nth(i);
		Space *dependentLps = sync->getDependentLps();
		stream << indentStr.str() << "if (threadState->isValidPpu(Space_" << dependentLps->getName();
		stream << ")) {\n";
		stream << indentStr.str() << '\t';
		sync->writeDescriptiveComment(stream, true);
		stream << indentStr.str() << "}\n";
	}

	// write the code for backword sync requirements within an if block
	if (backwardSyncs->NumElements() > 0) {
		stream << indentStr.str() << "if (repeatIteration > 0) {\n";
		for (int i = 0; i < backwardSyncs->NumElements(); i++) {
			SyncRequirement *sync = backwardSyncs->Nth(i);
			Space *dependentLps = sync->getDependentLps();
			stream << indentStr.str() << "\tif (threadState->isValidPpu(Space_" << dependentLps->getName();
			stream << ")) {\n";
			stream << indentStr.str() << "\t\t";
			sync->writeDescriptiveComment(stream, true);
			stream << indentStr.str() << "\t}\n";
		}
		stream << indentStr.str() << "}\n";
	}

	// finally deactive all sync dependencies as they are already been taken care of here	
	for (int i = 0; i < syncDependencies->NumElements(); i++) {
		SyncRequirement *sync = syncDependencies->Nth(i);
		sync->deactivate();
	}
}

List<const char*> *CompositeStage::getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel) {
	
	List<const char*> *arcNameList = new List<const char*>;
	List<const char*> *ownList = FlowStage::getAllOutgoingDependencyNamesAtNestingLevel(nestingLevel);
	if (ownList != NULL) {
		arcNameList->AppendAll(ownList);
	}

	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		List<const char*> *nestedList = stage->getAllOutgoingDependencyNamesAtNestingLevel(nestingLevel);
		if (nestedList != NULL) {
			for (int j = 0; j < nestedList->NumElements(); j++) {
				const char *currentArcName = nestedList->Nth(j);
				bool found = false;
				for (int k = 0; k < arcNameList->NumElements(); k++) {
					if (strcmp(arcNameList->Nth(k), currentArcName) == 0) {
						found = true;
						break;
					}
				}
				if (!found) {
					arcNameList->Append(currentArcName);
				}
			}
		}	
	}
	return arcNameList;
}

List<DependencyArc*> *CompositeStage::getAllTaskDependencies() {
	List<DependencyArc*> *list = new List<DependencyArc*>;
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		list->AppendAll(stage->getAllTaskDependencies());
	}
	return list;
}

List<const char*> *CompositeStage::getVariablesNeedingCommunication(int segmentedPPS) {
	List<const char*> *varList = new List<const char*>;
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		List<const char*> *stageVarList = stage->getVariablesNeedingCommunication(segmentedPPS);
		if (stageVarList == NULL) continue;
		for (int j = 0; j < stageVarList->NumElements(); j++) {
			const char *varName = stageVarList->Nth(j);
			if (!string_utils::contains(varList, varName)) {
				varList->Append(varName);
			}
		}
		delete stageVarList;
	}
	return varList;
}

List<CommunicationCharacteristics*> *CompositeStage::getCommCharacteristicsForSyncReqs(int segmentedPPS) {
	
	List<CommunicationCharacteristics*> *commCharList = new List<CommunicationCharacteristics*>;
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		List<CommunicationCharacteristics*> *stageCommList 
				= stage->getCommCharacteristicsForSyncReqs(segmentedPPS);
		if (stageCommList != NULL) {
			commCharList->AppendAll(stageCommList);
		}
		delete stageCommList;
	}
	return commCharList;
}

void CompositeStage::declareSynchronizationCounters(std::ofstream &stream, int indentation, int nestingIndex) {

	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	List<const char*> *counterNameList = getAllOutgoingDependencyNamesAtNestingLevel(nestingIndex);
	if (counterNameList != NULL && counterNameList->NumElements() > 0) {
		stream << std::endl << indent.str() << "// declaration of synchronization counter variables\n";
		for (int i = 0; i < counterNameList->NumElements(); i++) {
			const char *counter = counterNameList->Nth(i);
			stream << indent.str() << "int " << counter << " = 0";
			stream << stmtSeparator;
		}
	}
}

List<SyncRequirement*> *CompositeStage::getUpdateSignalsOfGroup(List<FlowStage*> *group) {
	List<SyncRequirement*> *syncList = new List<SyncRequirement*>;
	for (int i = 0; i < group->NumElements(); i++) {
		FlowStage *stage = group->Nth(i);
		List<SyncRequirement*> *activeStageSignals 
				= stage->getAllSyncRequirements()->getAllNonSignaledSyncReqs();
		if (activeStageSignals != NULL) {
			syncList->AppendAll(activeStageSignals);
		}
	}
	return syncList;
}

void CompositeStage::generateDataSendsForGroup(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *commRequirements) {
	
	std::ostringstream indentStr;
	for (int i = 0; i < indentation; i++) indentStr << indent;

	if (commRequirements->NumElements() > 0) {
		stream << std::endl << indentStr.str() << "// communicating updates\n";
	}

	// iterate over all the update signals
	for (int i = 0; i < commRequirements->NumElements(); i++) {
		
		SyncRequirement *currentComm = commRequirements->Nth(i);
		int commIndex = currentComm->getIndex();
		const char *counterVarName = currentComm->getDependencyArc()->getArcName();
		
		// check if the current PPU is a valid candidate for signaling update
		Space *signalingLps = currentComm->getDependencyArc()->getSource()->getSpace();
		stream << indentStr.str() << "if (threadState->isValidPpu(Space_" << signalingLps->getName();
		stream << ")) {\n";
		
		// retrieve the communicator for this dependency
		stream << indentStr.str() << indent;
		stream << "Communicator *communicator = threadState->getCommunicator(\"";
		stream << currentComm->getDependencyArc()->getArcName() << "\")" << stmtSeparator;
		stream << indentStr.str() << indent << "if (communicator != NULL) {\n";
		
		// Check if the current communication is conditional, i.e., it only gets signaled by threads that
		// executed a certain execution flow-stage. In that case, there will be a counter variable set to a
		// non-zero value. For communication signals issued by compiler injected sync-stages, there will be
		// no such counter variable. 
		bool needCounter = currentComm->getCounterRequirement();
		
		if (needCounter) {
			// If the counter variable for the sync has been updated then the current PPU controller has 
			// data to send so it should indicate that fact in its call to the communicator. Otherwise
			// it should only report that it has reached this particular execution point.
			stream << indentStr.str() << doubleIndent << "if (" << counterVarName << " > 0) {\n";
			stream << indentStr.str() << tripleIndent << "communicator->send(";
			stream << "REQUESTING_COMMUNICATION" << paramSeparator;
			stream << "commCounter" << commIndex <<  ")" << stmtSeparator;
			stream << indentStr.str() << doubleIndent << "} else communicator->send(";
			stream << "PASSIVE_REPORTING" << paramSeparator;
			stream << "commCounter" << commIndex << ")" << stmtSeparator;
		} else {
			stream << indentStr.str() << doubleIndent << "communicator->send(";
			stream << "REQUESTING_COMMUNICATION" << paramSeparator;
			stream << "commCounter" << commIndex <<  ")" << stmtSeparator;
		}

		stream << indentStr.str() << indent << "}\n";
		
		// then reset the counter
		if (needCounter) {	 
			stream << indentStr.str() << indent << counterVarName << " = 0" << stmtSeparator;
		}
		stream << indentStr.str() << "}\n";
	}
}

void CompositeStage::generateSignalCodeForGroupTransitions(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *syncRequirements) {

	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	if (syncRequirements->NumElements() > 0) {
		stream << std::endl << indent.str() << "// issuing synchronization signals\n";
	}

	// iterate over all the synchronization signals
	for (int i = 0; i < syncRequirements->NumElements(); i++) {
		SyncRequirement *currentSync = syncRequirements->Nth(i);
		const char *counterVarName = currentSync->getDependencyArc()->getArcName();
		// check if the concerned update did take place
		stream << indent.str() << "if (" << counterVarName << " > 0 && ";
		// also check if the current PPU is a valid candidate for signaling update
		Space *signalingLps = currentSync->getDependencyArc()->getSource()->getSpace();
		stream << "threadState->isValidPpu(Space_" << signalingLps->getName();
		stream << ")) {\n";
		// then signal synchronization (which is now just writing a comment)
		stream << indent.str() << '\t';
		currentSync->writeDescriptiveComment(stream, false);
		// then reset the counter	 
		stream << indent.str() << '\t';
		stream << counterVarName << " = 0" << stmtSeparator;
		stream << indent.str() << "}\n";
	}
}

void CompositeStage::setReactivatorFlagsForSyncReqs() {
	
	// set reactivator flags for all sync primitives operating on this composite stage level
	FlowStage::setReactivatorFlagsForSyncReqs();

	// then set reactivator flags for nested computations 
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->setReactivatorFlagsForSyncReqs(); 
	} 
}

void CompositeStage::generateCodeForWaitingForReactivation(std::ofstream &stream, int indentation, 
		List<SyncRequirement*> *syncRequirements) {
	
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	List<DependencyArc*> *reactivatorSyncs = new List<DependencyArc*>;
	for (int i = 0; i < syncRequirements->NumElements(); i++) {
		SyncRequirement *sync = syncRequirements->Nth(i);
		DependencyArc *arc = sync->getDependencyArc();
		if (arc->isReactivator()) {
			reactivatorSyncs->Append(arc);	
		}
	}

	if (reactivatorSyncs->NumElements() > 0) {
		stream << std::endl << indent.str() << "// waiting for signal clearance from readers\n";
	}

	for (int i = 0; i < reactivatorSyncs->NumElements(); i++) {
		DependencyArc *arc = reactivatorSyncs->Nth(i);
		Space *signalingLps = arc->getSignalSrc()->getSpace();
		stream << indent.str() << "if (threadState->isValidPpu(Space_";
		stream << signalingLps->getName();
		stream << ")) {\n";
		stream << indent.str() << '\t';
		stream << "// waiting the signal for " << arc->getArcName() << " to be cleared\n";
		stream << indent.str() << "}\n";
	}			
}

void CompositeStage::generateCodeForReactivatingDataModifiers(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *syncDependencies) {
	
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	List<DependencyArc*> *reactivatorSyncs = new List<DependencyArc*>;
	for (int i = 0; i < syncDependencies->NumElements(); i++) {
		SyncRequirement *sync = syncDependencies->Nth(i);
		DependencyArc *arc = sync->getDependencyArc();
		if (arc->isReactivator()) {
			reactivatorSyncs->Append(arc);	
		}
	}

	if (reactivatorSyncs->NumElements() > 0) {
		stream << std::endl << indent.str() << "// sending clearance signals to writers\n";
	}

	for (int i = 0; i < reactivatorSyncs->NumElements(); i++) {
		DependencyArc *arc = reactivatorSyncs->Nth(i);
		Space *sinkLps = arc->getSignalSink()->getSpace();
		stream << indent.str() << "if (threadState->isValidPpu(Space_";
		stream << sinkLps->getName();
		stream << ")) {\n";
		stream << indent.str() << '\t';
		stream << "// sending clearance for " << arc->getArcName() << " signal\n";
		stream << indent.str() << "}\n";
	}			
}

void CompositeStage::genSimplifiedWaitingForReactivationCode(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *syncRequirements) {

	std::string stmtSeparator = ";\n";	
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	if (syncRequirements->NumElements() > 0) {
		stream << std::endl << indent.str();
		stream << "// barriers to ensure all readers have finished reading last update\n";
	}

	for (int i = 0; i < syncRequirements->NumElements(); i++) {
		SyncRequirement *sync = syncRequirements->Nth(i);
		Space *syncSpanLps = sync->getSyncSpan();
		stream << indent.str() << "if (threadState->isValidPpu(Space_";
		stream << syncSpanLps->getName();
		stream << ")) {\n";	
		stream << indent.str() << '\t';
		stream << "threadSync->" << sync->getReverseSyncName() << "->wait()";
		stream << stmtSeparator;
		stream << indent.str() << "}\n";
	}
}

void CompositeStage::genSimplifiedSignalsForGroupTransitionsCode(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *syncRequirements) {

	std::string stmtSeparator = ";\n";
	std::string paramSeparator = ", ";
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';

	if (syncRequirements->NumElements() > 0) {
		stream << std::endl << indent.str() << "// resolving synchronization dependencies\n";
	}

	// iterate over all the synchronization signals and then issue signals and waits in a lock-step fasion
	for (int i = 0; i < syncRequirements->NumElements(); i++) {
		
		SyncRequirement *currentSync = syncRequirements->Nth(i);
		const char *counterVarName = currentSync->getDependencyArc()->getArcName();
	
		// Check if the current synchronization is conditional, i.e., it only gets signaled by threads that
		// executed a certain execution flow-stage. In that case, there will be a counter variable set to a
		// non-zero value. For synchronization signal issued by compiler injected sync-stages, there will be
		// no such counter variable. 
		bool needCounter = currentSync->getCounterRequirement();
		
		stream << indent.str() << "if (";
		// check if the concerned update did take place
		if (needCounter) {
			stream << counterVarName << " > 0 && ";
		}
		// also check if the current PPU is a valid candidate for signaling update
		FlowStage *sourceStage = currentSync->getDependencyArc()->getSource();
		Space *signalingLps = sourceStage->getSpace();
		stream << "threadState->isValidPpu(Space_" << signalingLps->getName();
		stream << ")) {\n";
		// then signal synchronization
		stream << indent.str() << '\t';
		stream << "threadSync->" << currentSync->getSyncName() << "->signal(";
		FlowStage *signalSource = currentSync->getDependencyArc()->getSignalSrc();
		if (signalSource->getRepeatIndex() > 0) stream << "repeatIteration";
		else stream << "0";
		stream << ")" << stmtSeparator;
		// then reset the counter
		if (needCounter) {	 
			stream << indent.str() << '\t';
			stream << counterVarName << " = 0" << stmtSeparator;
		}

		// the waiting is in an else block coupled with the signaling if block as the current implementation
		// of synchronization primitives does not support the PPU (or PPUs) in the signaling block to also be
		// among the list of waiting PPUs.  
		stream << indent.str() << "} else if (";
		Space *syncSpanLps = currentSync->getSyncSpan();
		FlowStage *waitingStage = currentSync->getWaitingComputation();
		stream << "threadState->isValidPpu(Space_" << syncSpanLps->getName();
		stream << ")) {\n";
		stream << indent.str() << '\t';
		stream << "threadSync->" << currentSync->getSyncName() << "->wait(";
		FlowStage *signalSink = currentSync->getDependencyArc()->getSignalSink();
		if (signalSink->getRepeatIndex() > 0) stream << "repeatIteration";
		else stream << "0";
		stream << ")" << stmtSeparator;
		stream << indent.str() << "}\n";
	}
}

void CompositeStage::initializeReductionResults(std::ofstream &stream, int indentation) {
	
	std::ostringstream indents;
	for (int i = 0; i < indentation; i++) indents << indent;

	stream << std::endl;
	stream << indents.str() << "// initializing thread-local reduction result variables\n";

	for (int i = 0; i < nestedReductions->NumElements(); i++) {
		ReductionMetadata *reduction = nestedReductions->Nth(i);
		const char *resultVar = reduction->getResultVar();
		Space *executingLps = reduction->getReductionExecutorLps();
		const char *execLpsName = executingLps->getName();
		stream << indents.str() << "if(threadState->isValidPpu(Space_" << execLpsName << ")) {\n";
		stream << indents.str() << indent;
		stream << "reduction::Result *" << resultVar << "Local = reductionResultsMap->";
		stream << "Lookup(\"" << resultVar << "\")" << stmtSeparator;
		stream << indents.str() << indent;
		stream << "ReductionPrimitive *rdPrimitive = rdPrimitiveMap->Lookup(\"";
		stream << resultVar << "\")" << stmtSeparator;
		stream << indents.str() << indent << "rdPrimitive->resetPartialResult(";
		stream << resultVar << "Local)" << stmtSeparator;	
		stream << indents.str() << "}\n";
	}

	stream << std::endl;	
}

void CompositeStage::executeFinalStepOfReductions(std::ofstream &stream, int indentation) {
	
	std::ostringstream indents;
	for (int i = 0; i < indentation; i++) indents << indent;

	stream << std::endl;
	stream << indents.str() << "// executing the final step of reductions\n";
	
	for (int i = 0; i < nestedReductions->NumElements(); i++) {
		ReductionMetadata *reduction = nestedReductions->Nth(i);
		if (!reduction->isSingleton()) {
			std::cout << "The current compiler still cannot handle non task-global reductions\n";
			std::exit(EXIT_FAILURE);
		}
		const char *resultVar = reduction->getResultVar();
		Space *executingLps = reduction->getReductionExecutorLps();
		const char *execLpsName = executingLps->getName();
		stream << indents.str() << "if(threadState->isValidPpu(Space_" << execLpsName << ")) {\n";
		stream << indents.str() << indent;
		stream << "reduction::Result *localResult = reductionResultsMap->";
		stream << "Lookup(\"" << resultVar << "\")" << stmtSeparator;
		stream << indents.str() << indent;
		stream << "void *target = &(taskGlobals->" << resultVar << ")" << stmtSeparator;
		stream << indents.str() << indent;
		stream << "ReductionPrimitive *rdPrimitive = rdPrimitiveMap->Lookup(\"";
		stream << resultVar << "\")" << stmtSeparator;
		stream << indents.str() << indent;
		stream << "rdPrimitive->reduce(localResult" << paramSeparator << "target)" << stmtSeparator;
		stream << indents.str() << "}\n";
	}
	stream << std::endl;
}

//------------------------------------------------- Repeat Cycle ------------------------------------------------------/

RepeatCycle::RepeatCycle(int index, Space *space, RepeatCycleType type, Expr *executeCond) 
		: CompositeStage(index, space, NULL) {
	this->type = type;
	this->repeatCond = executeCond;
}

// Here we do the dependency analysis twice to take care of any new dependencies that may arise due to the
// return from the last stage of repeat cycle to the first one. Note that the consequence of this double iteration
// may seems to be addition of superfluous dependencies when we have nesting of repeat cycles. However that should
// not happen as there is a redundancy checking mechanism in place where we add dependency arcs to flow stages. So
// unwanted arcs will be dropped off. 
void RepeatCycle::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
	CompositeStage::performDependencyAnalysis(hierarchy);	
	FlowStage::performDependencyAnalysis(repeatConditionAccessMap, hierarchy);
	CompositeStage::performDependencyAnalysis(hierarchy);	
}

void RepeatCycle::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
	if (repeatConditionAccessMap != NULL) {
		for (int i = 0; i < envAccessList->NumElements(); i++) {
			VariableAccess *envAccessLog = envAccessList->Nth(i);
			const char *varName = envAccessLog->getName();
			VariableAccess *stageAccessLog = repeatConditionAccessMap->Lookup(varName);
			if (stageAccessLog != NULL) {
				envAccessLog->mergeAccessInfo(stageAccessLog);
			}
		}		
	}
	CompositeStage::fillInTaskEnvAccessList(envAccessList);
}

void RepeatCycle::prepareTaskEnvStat(TaskEnvStat *taskStat, Hashtable<VariableAccess*> *accessMap) {
	if (repeatConditionAccessMap != NULL) {
		FlowStage::prepareTaskEnvStat(taskStat, repeatConditionAccessMap);
	}
	CompositeStage::prepareTaskEnvStat(taskStat);
}

void RepeatCycle::performEpochUsageAnalysis() {
	CompositeStage::performEpochUsageAnalysis();
	FlowStage::CurrentFlowStage = this;
	if (repeatCond != NULL) repeatCond->setEpochVersions(space, 0);
}

void RepeatCycle::calculateLPSUsageStatistics() {
	Iterator<VariableAccess*> iterator = repeatConditionAccessMap->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iterator.GetNextValue()) != NULL) {
		if (!accessLog->isContentAccessed()) continue;
		const char *varName = accessLog->getName();
		DataStructure *structure = space->getLocalStructure(varName);
		AccessFlags *accessFlags = accessLog->getContentAccessFlags();
		LPSVarUsageStat *usageStat = structure->getUsageStat();
		// As repeat condition is supposed to be evaluated multiple times, any data structure been used
		// here should be marked as multiple access. Thus, we add access information twice.
		if (accessFlags->isRead() || accessFlags->isWritten()) {
			usageStat->addAccess();
			usageStat->addAccess();
		}
	}
	// Two calls have been made for evaluating the usage statistics in nested computation stages assuming that
	// a repeat cycle will in most cases executes at least twice.
	CompositeStage::calculateLPSUsageStatistics();
	CompositeStage::calculateLPSUsageStatistics();
}

List<DependencyArc*> *RepeatCycle::getAllTaskDependencies() {
	List<DependencyArc*> *list = CompositeStage::getAllTaskDependencies();
	list->AppendAll(FlowStage::getAllTaskDependencies());
	return list;
}

int RepeatCycle::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {
	
	this->index = currentIndex;
	this->groupNo = currentGroupNo;
	this->repeatIndex = currentRepeatCycle;

	int nextIndex = currentIndex + 1;
	int nextRepeatIndex = currentRepeatCycle + 1;

	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		nextIndex = stage->assignIndexAndGroupNo(nextIndex, this->index, nextRepeatIndex);
	}
	return nextIndex;
	
}

void RepeatCycle::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	std::string stmtSeparator = ";\n";

	// if it is not a sub-partition repeat block then we have to add a while or for loop depending on the
	// condition
	if (type == Conditional_Repeat) {
		
		std::ostringstream indent;
		for (int i = 0; i < indentation; i++) indent << '\t';
		
		// create a scope for repeat loop
		stream << std::endl << indent.str() << "{ // scope entrance for repeat loop\n";

		// declare a repeat iteration number tracking variable
		stream << indent.str() << "int repeatIteration = 0" << stmtSeparator;

		// get the name of the lpu for the execution LPS
		std::ostringstream lpuName;
		lpuName << "space" << space->getName() << "Lpu->";

		// If the repeat condition involves accessing metadata of some task global array then we need 
		// to create local copies of its metadata so that name transformer can work properly 
        	if (isLpsDependent()) {
			List<const char*> *localArrays = filterInArraysFromAccessMap(repeatConditionAccessMap);
			for (int i = 0; i < localArrays->NumElements(); i++) {
        			const char *arrayName = localArrays->Nth(i);
        			ArrayDataStructure *array = (ArrayDataStructure*) space->getStructure(arrayName);
        			int dimensions = array->getDimensionality();
        			stream << indent.str() << "Dimension ";
                		stream  << arrayName << "PartDims[" << dimensions << "];\n";
        			stream << indent.str() << "Dimension ";
                		stream  << arrayName << "StoreDims[" << dimensions << "];\n";
                		for (int j = 0; j < dimensions; j++) {
                			stream << indent.str();
               				stream << arrayName << "PartDims[" << j << "] = " << lpuName.str();
                        		stream << arrayName << "PartDims[" << j << "].partition;\n";
                			stream << indent.str();
               				stream << arrayName << "StoreDims[" << j << "] = " << lpuName.str();
                        		stream << arrayName << "PartDims[" << j << "].storage;\n";
        			}
        		}
		}

		// update the name transformer for probable array access within repeat condition
		ntransform::NameTransformer::transformer->setLpuPrefix(lpuName.str().c_str());
		
		// if the repeat condition is a logical expression then it is a while loop in the source code
		if (dynamic_cast<LogicalExpr*>(repeatCond) != NULL) {
			std::ostringstream condition;
			repeatCond->translate(condition, indentation, 0, space);
			stream << indent.str() << "while (" << condition.str() << ") {\n";
			// declare all synchronization counter variables here that will be updated inside 
			// repeat loop 
			declareSynchronizationCounters(stream, indentation + 1, this->repeatIndex + 1);
			// invoke the repeat body
			CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);
			// increase the loop iteration counter
			stream << indent.str() << "\trepeatIteration++" << stmtSeparator;
			stream << indent.str() << "}\n";
		// otherwise it is a for loop based on a range condition that we need to translate
		} else {
			// translate the range expression into a for loop
			RangeExpr *rangeExpr = dynamic_cast<RangeExpr*>(repeatCond);
			std::ostringstream rangeLoop;
			rangeExpr->generateLoopForRangeExpr(rangeLoop, indentation, space);
			stream << rangeLoop.str();
			// declare all synchronization counter variables here that will be updated inside 
			// repeat loop 
			declareSynchronizationCounters(stream, indentation + 1, this->repeatIndex + 1);
			// translate the repeat body
			CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);
			// increase the loop iteration counter
			stream << indent.str() << "\trepeatIteration++" << stmtSeparator;
			// close the range loop
			stream << indent.str() << "}\n";
		}
		// exit the scope created for the repeat loop 
		stream << indent.str() << "} // scope exit for repeat loop\n";
	} else {
		// declare any synchronization counter variable needed inside the repeat loop
		declareSynchronizationCounters(stream, indentation, this->repeatIndex + 1);
		
		// TODO probably we need to maintain a repeat iteration counter in this case two. Then we should
		// change this straightforward superclass's code execution strategy. We should investigate this
		// in the future 
		CompositeStage::generateInvocationCode(stream, indentation, containerSpace);
	}
}

bool RepeatCycle::isLpsDependent() {
	VariableAccess *accessLog;
	Iterator<VariableAccess*> iterator = repeatConditionAccessMap->GetIterator();
	while ((accessLog = iterator.GetNextValue()) != NULL) {
		if (!(accessLog->isContentAccessed() 
			|| (accessLog->isMetadataAccessed() 
				&& accessLog->isLocalAccess()))) continue;
		const char *varName = accessLog->getName();
		DataStructure *structure = space->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) return true;
	}
	return false;
}

void RepeatCycle::setLpsExecutionFlags() {
	CompositeStage::setLpsExecutionFlags();
	if (isLpsDependent()) {
		space->flagToExecuteCode();
	}
}

