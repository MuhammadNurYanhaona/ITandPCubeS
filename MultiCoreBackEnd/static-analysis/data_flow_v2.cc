#include "data_access.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "data_flow.h"
#include "../semantics/task_space.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"

#include <iostream>
#include <fstream>
#include <sstream>

//-------------------------------------------------- Flow Stage ----------------------------------------------------------/

FlowStage::FlowStage(int index, Space *space, Expr *executeCond) {
	this->index = index;
	this->space = space;
	this->executeCond = executeCond;
	this->accessMap = new Hashtable<VariableAccess*>;
	this->dataDependencies = new DataDependencies();
	this->name = NULL;
}

void FlowStage::mergeAccessMapTo(Hashtable<VariableAccess*> *destinationMap) {
	Stmt::mergeAccessedVariables(destinationMap, accessMap);
}

void FlowStage::addAccessInfo(VariableAccess *accessLog) {
	const char *varName = accessLog->getName();
	if (accessMap->Lookup(varName) != NULL) accessMap->Lookup(varName)->mergeAccessInfo(accessLog);
	else accessMap->Enter(varName, accessLog, true);
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForSpaceInIndexLimit(Space *space,
                        List<FlowStage*> *stageList, int startIndex, int endIndex,
                        bool includeMentionedSpace) {
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
	for (int i = startIndex; i <= endIndex; i++) {
		FlowStage *stage = stageList->Nth(i);
		Space *stageSpace = stage->getSpace();
		if (stageSpace->isParentSpace(space)) {
			Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
		} else if (stageSpace == space && includeMentionedSpace) {
			Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
		}
	}
	return accessLogs;
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForReturnToSpace(Space *space,
                        List<FlowStage*> *stageList, int endIndex) {
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
	for (int i = endIndex; i >= 0; i--) {
		FlowStage *stage = stageList->Nth(i);
		Space *stageSpace = stage->getSpace();
		if (!stageSpace->isParentSpace(space)) break;
		Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
	}
	return accessLogs;	
}

void FlowStage::print(int indent) {
	for (int i = 0; i < indent; i++) printf("\t");
	printf("Flow Stage: %s (Space: %s)", name, space->getName());
	if (executeCond != NULL) printf(" Conditionally Executed");
	printf("\n");
	Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess* accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
        	accessLog->printAccessDetail(indent + 1);
        }
	dataDependencies->print(indent + 1);
}

void FlowStage::performDependencyAnalysis(Hashtable<VariableAccess*> *accessLogs, PartitionHierarchy *hierarchy) {
	
	LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessLogs->GetIterator();
        VariableAccess *accessLog;

        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
                                DependencyArc *arc = new DependencyArc(modifier, this, accessLog->getName());
				arc->deriveSyncAndCommunicationRoots(hierarchy);
                                dataDependencies->addIncomingArcIfNotExists(arc);
                        }
                }
                if (accessLog->isModified()) {
                        modifierPanel->setLastModifierOfVar(this, accessLog->getName());
                }
        }
}

bool FlowStage::isDataModifierRelevant(FlowStage *modifier) {
	SyncStage *syncStage = dynamic_cast<SyncStage*>(modifier);
	if (syncStage == NULL || !syncStage->isLoaderSync()) return true;
	return (this->space == syncStage->space || this->space->isParentSpace(syncStage->space));
}

//-------------------------------------------------- Sync Stage ---------------------------------------------------------/

SyncStage::SyncStage(Space *space, SyncMode mode, SyncStageType type) : FlowStage(0, space, NULL) {
	
	this->mode = mode;	
	this->type = type;
	
	if (mode == Load || mode == Load_And_Configure) {
		this->name = "\"Data Loader Sync\"";
	} else if (mode == Ghost_Region_Update) {
		this->name = "\"Ghost Region Updater Sync\"";
	} else {
		this->name = "\"Data Restorer Sync\"";
	}
}

int SyncStage::populateAccessMap(List<VariableAccess*> *accessLogs, 
		bool filterOutNonReads, bool filterOutNonWritten) {
	int count = 0;
	for (int i = 0; i < accessLogs->NumElements(); i++) {
                VariableAccess *accessLog = accessLogs->Nth(i);
                if (accessLog->isContentAccessed()) {
			if (filterOutNonReads 
				&& !(accessLog->getContentAccessFlags()->isRead()
					|| accessLog->getContentAccessFlags()->isReduced())) continue;
			if (filterOutNonWritten && !accessLog->getContentAccessFlags()->isWritten()) continue;
			
			DataStructure *structure = space->getLocalStructure(accessLog->getName());
			ArrayDataStructure *arrayStruct = dynamic_cast<ArrayDataStructure*>(structure);
			bool scalar = (arrayStruct == NULL);
			
                        VariableAccess *syncLog = new VariableAccess(accessLog->getName());
                        syncLog->markContentAccess();
                        if (accessLog->getContentAccessFlags()->isWritten()) {
				syncLog->getContentAccessFlags()->flagAsWritten();
			} else if (!scalar) syncLog->getContentAccessFlags()->flagAsWritten();

			if (mode == Load_And_Configure) {
				syncLog->markMetadataAccess();
				syncLog->getMetadataAccessFlags()->flagAsWritten();
			}

                        syncLog->getContentAccessFlags()->flagAsRead();
                        addAccessInfo(syncLog);
                        count++;
                }
	}
	return count;
}

void SyncStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	// TODO need to provide an accurate implementation. Now we are just printing a comment
	for (int i = 0; i < indentation; i++) stream << "\t";
	stream << "//this is a comment correspond to a sync stage\n";	
}

//------------------------------------------------ Execution Stage -------------------------------------------------------/

ExecutionStage::ExecutionStage(int index, Space *space, Expr *executeCond) 
		: FlowStage(index, space, executeCond) {
	this->code = NULL;
	this->scope = NULL;
}

void ExecutionStage::setCode(List<Stmt*> *stmtList) {
	this->code = new StmtBlock(stmtList);
}

void ExecutionStage::translateCode(std::ofstream &stream) {

	// declare all local variables found in the scope
        Iterator<Symbol*> iterator = scope->get_local_symbols();
        Symbol *symbol;
	bool first = true;
        while ((symbol = iterator.GetNextValue()) != NULL) {
		if (first) {
			stream << "\n\t// declare local variables of this compute stage\n"; 
			first = false;
		}
                VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                if (variable == NULL) continue;
                Type *type = variable->getType();
                const char *name = variable->getName();
                stream << "\t" << type->getCppDeclaration(name) << ";\n";
        }

        // translate statements into C++ code
	stream <<  "\n\t//----------------------- Computation Begins --------------------------------\n";
	std::ostringstream codeStream;
	code->generateCode(codeStream, 1, space);
	stream << codeStream.str();
	stream <<  "\n\t//------------------------ Computation Ends ---------------------------------\n";
}

void ExecutionStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	// write the indent
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	std::ostringstream nextIndent;
	nextIndent << indent.str();

	// if this is not a group entry execution stage then add a condition ensure that a PPU with only
	// valid ID corresponding to this stage's LPS should go on executing the code
	if (!isGroupEntry()) {
		nextIndent << '\t';
		stream << indent.str() << "if (threadState.isValidPpu(Space_" << space->getName();
		stream << ")) {\n";
	}
	// invoke the related method with current LPU parameter
	stream << nextIndent.str() << "// invoking user computation\n";
	stream << nextIndent.str();
	stream << name << "(*space" << space->getName() << "Lpu, ";
	// along with other default arguments
	nextIndent << "\t\t";
	stream << '\n' << nextIndent.str() << "arrayMetadata,";
	stream << '\n' << nextIndent.str() << "taskGlobals,";
	stream << '\n' << nextIndent.str() << "threadLocals, partition);\n";

	// close the if condition if applicable
	if (!isGroupEntry()) {
		stream << indent.str() << "}\n";
	}
}

// An execution stage is a group entry if it contain any reduction operation
bool ExecutionStage::isGroupEntry() {
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
	VariableAccess *access;
	while ((access = iterator.GetNextValue()) != NULL) {
		if (access->isContentAccessed() && access->getContentAccessFlags()->isReduced()) {
			return true;
		}
	}
	return false;
}

//------------------------------------------------ Composite Stage -------------------------------------------------------/

CompositeStage::CompositeStage(int index, Space *space, Expr *executeCond) : FlowStage(index, space, executeCond) {
	stageList = new List<FlowStage*>;	
}

void CompositeStage::addStageAtBeginning(FlowStage *stage) {
	stageList->InsertAt(stage, 0);
}

void CompositeStage::addStageAtEnd(FlowStage *stage) {
	stageList->Append(stage);
	SyncStage *syncStage = dynamic_cast<SyncStage*>(stage);
	if (syncStage != NULL) {
		syncStage->setIndex(stageList->NumElements());
	}
}

void CompositeStage::insertStageAt(int index, FlowStage *stage) {
	stageList->InsertAt(stage, index);
	stage->setIndex(index);
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
				// new space is higher in the space hierarchy; so an exit from the old space 
				// should be recorded along with an entry to the new space
				SpaceEntryCheckpoint *oldCheckpoint = SpaceEntryCheckpoint::getCheckpoint(oldSpace);
				SyncStage *oldEntrySyncStage = oldCheckpoint->getEntrySyncStage();
				Hashtable<VariableAccess*> *accessLogs = getAccessLogsForSpaceInIndexLimit(oldSpace,
                        			stageList, oldCheckpoint->getStageIndex(), 
						nextStageIndex - 1, true);
				// If there is an entry sync stage for the old space then we need to populate
				// its accessMap currectly.
				if (oldEntrySyncStage != NULL) {
					SyncStageGenerator::populateAccessMapOfEntrySyncStage(oldEntrySyncStage, accessLogs);
				}
				// generate and add to the list all possible sync stages that are required due
				// to the exit from the old space
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
				// Old space is higher in the space hierarchy; so an entry to the new space
				// should be recorded. The entry sync stage here, if present, is just a place holder.
				// Later on during exit its accessLog is filled with appropriate data
				SyncStage *entrySyncStage = SyncStageGenerator::generateEntrySyncStage(newSpace);
				SpaceEntryCheckpoint *checkpoint = 
						SpaceEntryCheckpoint::addACheckpointIfApplicable(newSpace, nextStageIndex);
				checkpoint->setEntrySyncStage(entrySyncStage);
				if (entrySyncStage != NULL) {
					addStageAtEnd(entrySyncStage);
				}
			} else if (oldSpace == newSpace) {
				// transition is made from some computation of the same space to another; we
				// may need to synchronize overlapping boundary regions of data structures
				FlowStage *lastStageOnSpace = getLastNonSyncStage();
				Hashtable<VariableAccess*>  *accessLogs = lastStageOnSpace->getAccessMap();
				SyncStage *reappearanceSync = 
						SyncStageGenerator::generateReappearanceSyncStage(newSpace, accessLogs);
				if (reappearanceSync != NULL) {
					addStageAtEnd(reappearanceSync);
				}
			} else {
				printf("We should never have a disjoint space transition chain\n");
				exit(1);
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
                // If there is an entry sync stage for the old space then we need to populate
                // its accessMap currectly.
                if (oldEntrySyncStage != NULL) {
                	SyncStageGenerator::populateAccessMapOfEntrySyncStage(oldEntrySyncStage, accessLogs);
                }
                // generate and add to the list all possible sync stages that are required due
                // to the exit from the old space
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

List<List<FlowStage*>*> *CompositeStage::getConsecutiveNonLPSCrossingStages() {
	
	List<List<FlowStage*>*> *stageGroups = new List<List<FlowStage*>*>;
	Space *currentSpace = stageList->Nth(0)->getSpace();
	List<FlowStage*> *currentGroup = new List<FlowStage*>;
	currentGroup->Append(stageList->Nth(0));

	for (int i = 1; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		// if the stage is executing in a different LPS then create a new group
		if (stage->getSpace() != currentSpace) {
			currentSpace = stage->getSpace();
			stageGroups->Append(currentGroup);
			currentGroup = new List<FlowStage*>;
			currentGroup->Append(stage);
		// otherwise add the stage in the current group
		} else {
			currentGroup->Append(stage);
		}
	}
	stageGroups->Append(currentGroup);
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

	// if their is an LPS transition due to entering this stage then create a while loop traversing LPUs
	// of newly entered LPS
	if (this->space != containerSpace) {
		const char *spaceName = space->getName();
		nextIndentation++;
		nextIndent << '\t';
		// create a new local scope for traversing LPUs of this new scope
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
		stream << indent.str() << "while((lpu = threadState.getNextLpu(";
		stream << "Space_" << spaceName << paramSeparator << "Space_" << containerSpace->getName();
		stream << paramSeparator << "space" << spaceName << "LpuId)) != NULL) {\n";
		// cast the common LPU variable to LPS specific LPU		
		stream << nextIndent.str() << "space" << spaceName << "Lpu = (Space" << spaceName;
		stream  << "_LPU*) lpu" << stmtSeparator;
	}
	
	// Iterate over groups of flow stages where each group executes within a single LPS. This scheme has the
	// consequence of generating LPU only one time for all stages of a group then execute all of them before
	// proceed to the next LPU 
	List<List<FlowStage*>*> *stageGroups = getConsecutiveNonLPSCrossingStages();
	
	for (int i = 0; i < stageGroups->NumElements(); i++) {
		List<FlowStage*> *currentGroup = stageGroups->Nth(i);
		Space *groupSpace = currentGroup->Nth(0)->getSpace();
		// if the LPS of the group is not the same of this one then we need to consider LPS entry, i.e.,
		// include a while loop for the group to traverse over the LPUs
		if (groupSpace != space) {
			// create a local composite flow stage to apply the logic of this function recursively
			CompositeStage *tempStage = new CompositeStage(0, groupSpace, NULL);
			tempStage->setStageList(currentGroup);
			tempStage->generateInvocationCode(stream, nextIndentation, space);
			delete tempStage;
		// otherwise the current LPU of this composite stage will suffice and we execute all the nested
		// stages one after one.	 
		} else {
			for (int j = 0; j < currentGroup->NumElements(); j++) {
				FlowStage *stage = currentGroup->Nth(j);
				stage->generateInvocationCode(stream, nextIndentation, space);
			}
		}
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
			stream << indent.str() << "threadState.removeIterationBound(Space_";
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
	for (int i = 1; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		if (stage->isGroupEntry()) return true;
	}
	List<List<FlowStage*>*> *stageGroups = getConsecutiveNonLPSCrossingStages();
	return (stageGroups->NumElements() > 1 || space != stageGroups->Nth(0)->Nth(0)->getSpace());
}

//------------------------------------------------- Repeat Cycle ------------------------------------------------------/

RepeatCycle::RepeatCycle(int index, Space *space, RepeatCycleType type, Expr *executeCond) 
		: CompositeStage(index, space, NULL) {
	this->type = type;
	this->repeatCond = executeCond;
}

void RepeatCycle::addSyncStagesOnReturn(List<FlowStage*> *stageList) {
	CompositeStage::addSyncStagesOnReturn(stageList);
	// For repeat cycles iterating on a space that has data structures with overlapped/ghost regions, extra
	// reappearance sync may be needed. Therefore, repeat cycle overrides this method and add an additional
	// clause.
        SyncStage *reappearanceSync = SyncStageGenerator::generateReappearanceSyncStage(space, accessMap);
        if (reappearanceSync != NULL) {
        	addStageAtEnd(reappearanceSync);
        }
}

// Here we do the dependency analysis twice to take care of any new dependencies that may arise due to the
// return from the last stage of repeat cycle to the first one. 
void RepeatCycle::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
	CompositeStage::performDependencyAnalysis(hierarchy);	
	FlowStage::performDependencyAnalysis(repeatConditionAccessMap, hierarchy);
	CompositeStage::performDependencyAnalysis(hierarchy);	
}

void RepeatCycle::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	std::string stmtSeparator = ";\n";
	
	// if it is not a sub-partition repeat block then we have to add a while or for loopd depending on the
	// condition
	if (type == Conditional_Repeat) {
		
		// translate the repeat condition
		std::ostringstream indent;
		for (int i = 0; i < indentation; i++) indent << '\t';
		
		// if the repeat condition is a logical expression then it is a while loop in the source code
		if (dynamic_cast<LogicalExpr*>(repeatCond) != NULL) {
			std::ostringstream condition;
			repeatCond->translate(condition, indentation, 0);
			stream << indent.str() << "while (" << condition.str() << ") {\n";
			CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);
			stream << indent.str() << "}\n";
		// otherwise it is a for loop based on a range condition that we need to translate
		} else {
			// create a scope for repeat loop
			stream << indent.str() << "{ // scope entrance for repeat loop\n";
			// translate the range expression into a for loop
			RangeExpr *rangeExpr = dynamic_cast<RangeExpr*>(repeatCond);
			std::ostringstream rangeLoop;
			rangeExpr->generateLoopForRangeExpr(rangeLoop, indentation, space);
			stream << rangeLoop.str();
			// translate the repeat body
			CompositeStage::generateInvocationCode(stream, indentation + 1, containerSpace);
			// close the range loop
			stream << indent.str() << "}\n";
			// exit the scope created for the repeat loop 
			stream << indent.str() << "} // scope exit for repeat loop\n";
		}
	} else {
		CompositeStage::generateInvocationCode(stream, indentation, containerSpace);
	}
}

