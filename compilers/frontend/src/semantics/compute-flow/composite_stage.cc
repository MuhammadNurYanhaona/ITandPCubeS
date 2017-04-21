#include "../computation_flow.h"
#include "../scope.h"
#include "../symbol.h"
#include "../task_space.h"
#include "../data_access.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_expr.h"
#include "../../syntax/ast_stmt.h"
#include "../../syntax/ast_task.h"
#include "../../static-analysis/sync_stat.h"
#include "../../static-analysis/reduction_info.h"
#include "../../static-analysis/sync_stage_implantation.h"
#include "../../codegen-helper/extern_config.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <iostream>
#include <fstream>

//-------------------------------------------------------- Composite Stage ------------------------------------------------------/

CompositeStage::CompositeStage(Space *space) : FlowStage(space) {
	this->stageList = new List<FlowStage*>;
}

void CompositeStage::setStageList(List<FlowStage*> *stageList) {
	this->stageList = stageList;
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->setParent(this);
	}
}

void CompositeStage::print(int indentLevel) {
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->print(indentLevel + 1);
	}
}

void CompositeStage::addStageAtBeginning(FlowStage *stage) {
        stageList->InsertAt(stage, 0);
        stage->setParent(this);
}

void CompositeStage::addStageAtEnd(FlowStage *stage) {
        stageList->Append(stage);
        stage->setParent(this);
}

void CompositeStage::insertStageAt(int index, FlowStage *stage) {
        stageList->InsertAt(stage, index);
        stage->setParent(this);
}

void CompositeStage::removeStageAt(int stageIndex) { stageList->RemoveAt(stageIndex); }

void CompositeStage::performDataAccessChecking(Scope *taskScope) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->performDataAccessChecking(taskScope);
	}
}

List<FlowStage*> *CompositeStage::swapStageList(List<FlowStage*> *argList) {
	List<FlowStage*> *oldList = this->stageList;
	this->stageList = argList;
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->setParent(this);
	}
	return oldList;
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

bool CompositeStage::isStageListEmpty() { return getLastNonSyncStage() == NULL; }

int CompositeStage::assignIndexAndGroupNo(int currentIndex, 
			int currentGroupNo, int currentRepeatCycle) {
        
	int nextIndex = FlowStage::assignIndexAndGroupNo(currentIndex, 
			currentGroupNo, currentRepeatCycle);
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                nextIndex = stage->assignIndexAndGroupNo(nextIndex, this->index, currentRepeatCycle);
        }
        return nextIndex;
}

void CompositeStage::makeAllLpsTransitionsExplicit() {
	
	List<FlowStage*> *currentMoverList = NULL;
	List<FlowStage*> *renewedStageList = new List<FlowStage*>;

	for (int i = 0; i < stageList->NumElements(); i++) {
		
		FlowStage *stage = stageList->Nth(i);
		Space *intermediateLps = FlowStage::getCommonIntermediateLps(this, stage);
		bool isSyncStage = (dynamic_cast<SyncStage*>(stage) != NULL);

		// sync stages are not pushed into intermediate LPS transition stages as they are compiler 
		// introduced stages with no code generation requirements
		if (intermediateLps == NULL || isSyncStage) {

			if (currentMoverList != NULL) {
				// create a new LPS transition stage with the mover list
				Space *commonLps = FlowStage::getCommonIntermediateLps(this,
						currentMoverList->Nth(0));
				LpsTransitionBlock *transitionContainer 
						= new LpsTransitionBlock(commonLps, this->space);
				transitionContainer->setStageList(currentMoverList);

				// insert the LPS transition container stage in current stage's nenewed list
				renewedStageList->Append(transitionContainer);
				
				// reset the current mover list
				currentMoverList = NULL;
			}
			// insert the current stage in the renewed list  
			renewedStageList->Append(stage);
		} else {
			if (currentMoverList == NULL) {
				currentMoverList = new List<FlowStage*>;
			}
			currentMoverList->Append(stage);
		}
		
		// let the recursive restructuring process move to nested stages
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
		if (compositeStage != NULL) {
			compositeStage->makeAllLpsTransitionsExplicit();
		}
	}

	// if the current mover list is not empty then do the processing of the remaining stages
	if (currentMoverList != NULL) {
		Space *commonLps = FlowStage::getCommonIntermediateLps(this, currentMoverList->Nth(0));
		LpsTransitionBlock *transitionContainer = new LpsTransitionBlock(commonLps, this->space);
		transitionContainer->setStageList(currentMoverList);
		renewedStageList->Append(transitionContainer);
	}

	// update the stage list of the current composite stage
	this->setStageList(renewedStageList);
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

void CompositeStage::implantSyncStagesInFlow(CompositeStage *containerStage, List<FlowStage*> *currStageList) {
	
	// prepare the current stage for sync-stage implantation process by creating a backup of the current
	// stage list
	List<FlowStage*> *oldStageList = swapStageList(new List<FlowStage*>);

	// invoke the sync-stage implantation function of the super-class to handle this stage's re-insertion to
	// the parent container stage
	if (containerStage != NULL) {
		FlowStage::implantSyncStagesInFlow(containerStage, currStageList);
	} else {
		// this is the terminal case: the beginning of the sync-stage implantation process
		currStageList->Append(this);
	}

	// then try to re-insert each flow-stage the current stage itself originally held one by one
	for (int i = 0; i < oldStageList->NumElements(); i++) {
		FlowStage *nestedStage = oldStageList->Nth(i);
		
		// the re-insertion process ensures that sync-stages are added before the nested stage as needed
		nestedStage->implantSyncStagesInFlow(this, currStageList);
	}

	// if the last stage of the sub-flow is assigned to a different LPS than the current stage then there 
	// might be a need for sync-stage implantation before exit; so take care of that
	addSyncStagesOnReturn(currStageList);	
}

void CompositeStage::populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress,
                Space *lps, bool includeLimiterLps) {

        if (space->isParentSpace(lps) || (lps == space && includeLimiterLps)) {
                Stmt::mergeAccessedVariables(accessMapInProgress, accessMap);
		for (int i = 0; i < stageList->NumElements(); i++) {
			FlowStage *stage = stageList->Nth(i);
			stage->populateAccessMapForSpaceLimit(accessMapInProgress, lps, includeLimiterLps);
		}	
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
        if (spaceTransitionChain == NULL) return;

        int nextStageIndex = stageList->NumElements();
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

			// If there is an entry sync stage for the old space then we need to populate its access map 
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
			// The entry sync stage here, if present, is just a place holder. Later on during the exit its 
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

                // if some data structures in the old space have overlapping boundary regions among their parts and 
		// some of those data structures have been modified, a ghost regions sync is needed that operate on the 
		// old space as overlapping boundaries should be synchronized at each space exit
                SyncStage *reappearanceSync =
                                SyncStageGenerator::generateReappearanceSyncStage(oldSpace, accessLogs);
                if (reappearanceSync != NULL) {
                        addStageAtEnd(reappearanceSync);
                }

                // generate and add to the list all possible sync stages that are required due to the exit from the old 
		// space
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

void CompositeStage::calculateLPSUsageStatistics() {
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->calculateLPSUsageStatistics();
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

void CompositeStage::setLpsExecutionFlags() {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->setLpsExecutionFlags();
        }
}

void CompositeStage::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->fillInTaskEnvAccessList(envAccessList);
        }
}

void CompositeStage::prepareTaskEnvStat(TaskEnvStat *taskStat) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->prepareTaskEnvStat(taskStat);
        }
}

void CompositeStage::populateReductionMetadata(PartitionHierarchy *lpsHierarchy) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->populateReductionMetadata(lpsHierarchy);
        }
}

// composite stages do not add their own lists of reduction metadata into the argument list 
// as their own lists have been created by extracting metadata from nested execution stages.
void CompositeStage::extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->extractAllReductionInfo(reductionInfos);
        }
}

List<ReductionMetadata*> *CompositeStage::upliftReductionInstrs() {
	
	List<ReductionMetadata*> *propagateInfoSet = new List<ReductionMetadata*>;

	int nestedStageCount = stageList->NumElements();
	for (int i = 0; i < nestedStageCount; i++) {
		
		// do recursive expansion of the uplifting process
		FlowStage *stage = stageList->Nth(i);
		List<ReductionMetadata*> *nestedInfoSet = stage->upliftReductionInstrs();
		
		// if the nested stage did not have any reduction or has already processed all nested reductions then
		// this stage can be skipped
		if (nestedInfoSet == NULL || nestedInfoSet->NumElements() == 0) continue;
		
		// determine what reductions should be sent upward
		// also determine the topmost LPS for any reduction that should execute within the confinement of this
		// composite stage
		Space *topmostReductionRootLps = NULL;
		for (int j = 0; j < nestedInfoSet->NumElements(); j++) {
			ReductionMetadata *metadata = nestedInfoSet->Nth(j);
                        Space *candidateLps = metadata->getReductionRootLps();
			if (this->space->isParentSpace(candidateLps)) {
				propagateInfoSet->Append(metadata);
			} else {
				if (topmostReductionRootLps == NULL 
						|| topmostReductionRootLps->isParentSpace(candidateLps)) {
					topmostReductionRootLps = candidateLps;
				}	
			}
		}
		// if all nested reductions should move upward then skip processing this stage further
		if (topmostReductionRootLps == NULL) continue;

		// if there are some reductions left that are not listed for propagating upward then create a reduction
		// boundary stage here
		ReductionBoundaryBlock *reductionBoundary = new ReductionBoundaryBlock(topmostReductionRootLps);
		reductionBoundary->addStageAtEnd(stage);

		// filter out reductions at the selected LPS from the nested stage to the reduction boundary
		List<ReductionMetadata*> *upliftedReductions = new List<ReductionMetadata*>;
		stage->filterReductionsAtLps(topmostReductionRootLps, upliftedReductions);
		reductionBoundary->assignReductions(upliftedReductions);

		// keep the recursive expansion process rolling within the reduction boundary in case the stage under
		// investigation has more reductions with root at some descendent LPS; note that this is done after we
		// filter out reductions that are intended for the reduction-boundary
		reductionBoundary->upliftReductionInstrs();	

		// if the reduction root is in a different LPS than the current LPS then put an LPS transition block 
		// around it
		CompositeStage *replacementStage = reductionBoundary;
		if (topmostReductionRootLps != this->space) {
			LpsTransitionBlock *transitionBlock 
					= new LpsTransitionBlock(topmostReductionRootLps, this->space);
			transitionBlock->addStageAtEnd(reductionBoundary);
			replacementStage = transitionBlock;
		}
		
		// swap the reduction boundary with the nested stage
		this->removeStageAt(i);
                this->insertStageAt(i, replacementStage);
	}

	// propagate all reductions that have root above the LPS of the current composite stage
	return propagateInfoSet;
}

void CompositeStage::filterReductionsAtLps(Space *reductionRootLps, List<ReductionMetadata*> *filteredList) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->filterReductionsAtLps(reductionRootLps, filteredList);
	}
}

FlowStage *CompositeStage::getLastAccessorStage(const char *varName) {
	FlowStage *lastAccessor = NULL;
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
		FlowStage *accessor = stage->getLastAccessorStage(varName);
		if (accessor != NULL) {
			lastAccessor = accessor;
		}
	}
	return lastAccessor;
}

void CompositeStage::validateReductions() {
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->validateReductions();	
	}
}

bool CompositeStage::hasExecutingCodeInDescendentLPSes() {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		Space *stagesLps = stage->getSpace();
		if (stagesLps->isParentSpace(this->space)) return true;
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
		if (compositeStage != NULL) {
			if (compositeStage->hasExecutingCodeInDescendentLPSes()) return true;
		}
	}
	return false;
}

Space *CompositeStage::getFurthestDescendentLpsWithinNestedFlow() {
	Space *lowermostLps = this->space;
	for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                Space *stagesLps = stage->getSpace();
                if (stagesLps->isParentSpace(lowermostLps)) {
			lowermostLps = stagesLps;
		}
                CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
                if (compositeStage != NULL) {
                	Space *nestedLowerLps = compositeStage->getFurthestDescendentLpsWithinNestedFlow();
			if (nestedLowerLps->isParentSpace(lowermostLps)) {
				lowermostLps = nestedLowerLps;
			}
                }
        }
	return lowermostLps;
}

void CompositeStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                stageList->Nth(i)->performDependencyAnalysis(hierarchy);
        }
}

void CompositeStage::analyzeSynchronizationNeeds() {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->analyzeSynchronizationNeeds();
        }
}

void CompositeStage::upliftSynchronizationDependencies() {

        // perform dependency uplifting in the nested composite stages
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
                if (compositeStage != NULL) {
                        compositeStage->upliftSynchronizationDependencies();
                }
        }

        // get the first and last flow index within the nested subflow to know the boundary of the current stage
        int beginIndex = stageList->Nth(0)->getIndex();
        int endIndex = getHighestNestedStageIndex();

        // then check stages and drag out any synchronization dependencies they have due to changes made in some
        // stages outside this composite stage as the dependencies of the current composite stage itself.
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                StageSyncDependencies *nestedDependencies = stage->getAllSyncDependencies();
                List<SyncRequirement*> *dependencyList = nestedDependencies->getDependencyList();
                for (int j = 0; j < dependencyList->NumElements(); j++) {
                        SyncRequirement *sync = dependencyList->Nth(j);
                        FlowStage *syncSource = sync->getDependencyArc()->getSignalSrc();
                        // if the source's index is within the nesting boundary then the dependency is internal
                        // otherwise, it is external and we have to pull it out
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

void CompositeStage::upliftSynchronizationNeeds() {

        // perform synchronization uplifting in the nested composite stages
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
                if (compositeStage != NULL) {
                        compositeStage->upliftSynchronizationNeeds();
                }
        }

        // get the first and last flow index within the nested subflow to know the boundary of the current stage
        int beginIndex = stageList->Nth(0)->getIndex();
        int endIndex = getHighestNestedStageIndex();

        // start extracting boundary crossing synchronizations out
	if (synchronizationReqs == NULL) {
        	synchronizationReqs = new StageSyncReqs(this);
	}

        // iterate over the nested stages and extract any boundary crossing synchronization found within and assign 
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
                                        // update the nesting index of the dependency arc so that it can be found 
					// later by its nesting index
                                        syncReq->getDependencyArc()->setNestingIndex(this->repeatIndex);
                                        // then update the signal source for the arc to indicate that it has been
                                        // lifted up
                                        syncReq->getDependencyArc()->setSignalSrc(this);
                                }
                        }
                }
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

void CompositeStage::printSyncRequirements(int indentLevel) {
	FlowStage::printSyncRequirements(indentLevel);
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->printSyncRequirements(indentLevel + 1);
        }
}

List<List<FlowStage*>*> *CompositeStage::getConsecutiveNonLPSCrossingStages() {

        List<List<FlowStage*>*> *stageGroups = new List<List<FlowStage*>*>;
        Space *currentSpace = stageList->Nth(0)->getSpace();
        List<FlowStage*> *currentGroup = new List<FlowStage*>;
        currentGroup->Append(stageList->Nth(0));

        for (int i = 1; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                RepeatControlBlock *repeatCycle = dynamic_cast<RepeatControlBlock*>(stage);
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
                        } else  currentGroup->Append(stage);
                }
        }
        if (currentGroup->NumElements() > 0) stageGroups->Append(currentGroup);
        return stageGroups;
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

void CompositeStage::retriveExternCodeBlocksConfigs(IncludesAndLinksMap *externConfigMap) {
        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                stage->retriveExternCodeBlocksConfigs(externConfigMap);
        }
}

