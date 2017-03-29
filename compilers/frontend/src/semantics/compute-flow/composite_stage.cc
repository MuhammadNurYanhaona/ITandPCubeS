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
#include "../../static-analysis/sync_stage_implantation.h"
#include "../../static-analysis/reduction_info.h"
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

void CompositeStage::makeAllLpsTransitionsExplicit() {
	
	List<FlowStage*> *currentMoverList = NULL;
	List<FlowStage*> *renewedStageList = new List<FlowStage*>;

	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		Space *intermediateLps = FlowStage::getCommonIntermediateLps(this, stage);
		if (intermediateLps == NULL) {
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

