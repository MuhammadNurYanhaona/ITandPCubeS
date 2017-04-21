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
#include "../../static-analysis/usage_statistic.h"
#include "../../static-analysis/task_env_stat.h"
#include "../../static-analysis/reduction_info.h"
#include "../../static-analysis/data_dependency.h"
#include "../../static-analysis/sync_stage_implantation.h"
#include "../../codegen-helper/communication_stat.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <iostream>
#include <fstream>
#include <sstream>

//---------------------------------------------------------- Flow Stage ---------------------------------------------------------/

FlowStage *FlowStage::CurrentFlowStage = NULL;

FlowStage::FlowStage(Space *space) {
        this->space = space;
        this->parent = NULL;
	this->location = NULL;
	this->accessMap = new Hashtable<VariableAccess*>;
	this->epochDependentVarList = new List<const char*>;
	this->dataDependencies = new DataDependencies();
	this->synchronizationReqs = NULL;
        this->syncDependencies = new StageSyncDependencies(this);
}

Hashtable<VariableAccess*> *FlowStage::getAccessMap() { return accessMap; }

Hashtable<VariableAccess*> *FlowStage::validateDataAccess(Scope *taskScope, Expr *activationCond, Stmt *code) {
	
	//------------------------------------------------------------------------first gather the access information

	TaskGlobalReferences *references = new TaskGlobalReferences(taskScope);
	Hashtable<VariableAccess*> *accessMap = new Hashtable<VariableAccess*>;
	if (activationCond != NULL) { 
		accessMap = activationCond->getAccessedGlobalVariables(references);
	}
	if (code != NULL) {
		Hashtable<VariableAccess*> *codeMap = code->getAccessedGlobalVariables(references);
		Stmt::mergeAccessedVariables(accessMap, codeMap);
	}

	//------------------------------------------------------------------------then validate the variable accesses
	
	// partitionable data structures (currently only arrays are partitionable) must be explicitly
        // mentioned in the partition section correspond to the LPS within which a computation stage executes. 
	// Scalar or non partitionable structures, on the other hand, are not specified explicitly. They get 
	// added to the LPS as they have been found to be used within an LPS's flow stages.
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                const char *name = accessLog->getName();
                if (space->isInSpace(name)) continue;
                VariableSymbol *symbol = (VariableSymbol*) taskScope->lookup(name);
                Type *type = symbol->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                if (arrayType != NULL && staticArray == NULL) {
                        ReportError::ArrayPartitionUnknown(location, name, space->getName());
                } else {
                        DataStructure *source = space->getStructure(name);
                        DataStructure *structure = new DataStructure(source);
                        space->addDataStructure(structure);
                }
        }

	return accessMap;
}

Space *FlowStage::getCommonIntermediateLps(FlowStage *container, FlowStage *contained) {
	
	Space *ancestorLps = container->getSpace();
        Space *descendentLps = contained->getSpace();
	if (ancestorLps == descendentLps 
			|| descendentLps->getParent() == ancestorLps) {
		return NULL;
	}

	Space *nextLevelLps = NULL;
	Space *currentLps = descendentLps->getParent();
	while (currentLps != ancestorLps) {
		nextLevelLps = currentLps;
		currentLps = currentLps->getParent();
	}
	return nextLevelLps;
}

int FlowStage::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {
        this->index = currentIndex;
        this->groupNo = currentGroupNo;
        this->repeatIndex = currentRepeatCycle;
        return currentIndex + 1;
}

const char *FlowStage::getName() {
	std::ostringstream name;
	name << "Flow Stage: [" << index;
	name << ", " << groupNo << ", " << repeatIndex<< "]";
	return strdup(name.str().c_str());
}

void FlowStage::implantSyncStagesInFlow(CompositeStage *containerStage, List<FlowStage*> *currStageList) {
	
	// before re-inserting the current flow stage in the container stage that was originally holding it, 
	// check if execution of earlier flow stages within the container necessitates some sync stage implant
	// before the current stage
	containerStage->addSyncStagesBeforeExecution(this, currStageList);
	
	// re-insert the current flow stage in the container 
	containerStage->addStageAtEnd(this);

	// add the current stage in the stage list to keep the reconstruction process rolling
	currStageList->Append(this);
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForSpaceInIndexLimit(Space *space,
                        List<FlowStage*> *stageList, 
			int startIndex, 
			int endIndex,
                        bool includeMentionedSpace) {

        Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
        for (int i = startIndex; i <= endIndex; i++) {
                FlowStage *stage = stageList->Nth(i);
		stage->populateAccessMapForSpaceLimit(accessLogs, space, includeMentionedSpace);	
        }
        return accessLogs;
}

Hashtable<VariableAccess*> *FlowStage::getAccessLogsForReturnToSpace(Space *space, 
		List<FlowStage*> *stageList, 
		int endIndex) {
        
	Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
        for (int i = endIndex; i >= 0; i--) {
                FlowStage *stage = stageList->Nth(i);
                Space *stageSpace = stage->getSpace();
                if (!stageSpace->isParentSpace(space)) break;
		stage->populateAccessMapForSpaceLimit(accessLogs, space, false);
        }
        return accessLogs;
}

void FlowStage::calculateLPSUsageStatistics() {
        
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iterator.GetNextValue()) != NULL) {
                if (!accessLog->isContentAccessed()) continue;
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getLocalStructure(varName);
                AccessFlags *accessFlags = accessLog->getContentAccessFlags();
                LPSVarUsageStat *usageStat = structure->getUsageStat();
                if (accessFlags->isRead() || accessFlags->isWritten()) {
                        usageStat->addAccess();
                }
                if (accessFlags->isReduced()) {
                        usageStat->flagReduced();
                }
        }
}



bool FlowStage::isLpsDependent() {
	VariableAccess *accessLog;
        Iterator<VariableAccess*> iterator = accessMap->GetIterator();
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

void FlowStage::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
        for (int i = 0; i < envAccessList->NumElements(); i++) {
                VariableAccess *envAccessLog = envAccessList->Nth(i);
                const char *varName = envAccessLog->getName();
                VariableAccess *stageAccessLog = accessMap->Lookup(varName);
                if (stageAccessLog != NULL) {
                        envAccessLog->mergeAccessInfo(stageAccessLog);
                }
        }
}

void FlowStage::prepareTaskEnvStat(TaskEnvStat *taskStat) {

        Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iterator.GetNextValue()) != NULL) {

                // if only the metadata of the object has been accessed then there is no need to track it
                if (!(accessLog->isRead() || accessLog->isModified())) continue;

                const char *varName = accessLog->getName();
                EnvVarStat *varStat = taskStat->getVariableStat(varName);

                // if there is no stat object for the variable access in the environment then the variable is not an 
		// environmental data structure
                if (varStat == NULL) continue;

                if (accessLog->isModified()) {
                        varStat->flagWriteOnLps(space);
                } else {
                        varStat->flagReadOnLps(space);
                }
        }
}

DataDependencies *FlowStage::getDataDependencies() { return dataDependencies; }

bool FlowStage::isDataModifierRelevant(FlowStage *modifier) {
        SyncStage *syncStage = dynamic_cast<SyncStage*>(modifier);
        if (syncStage == NULL || !syncStage->isLoaderSync()) return true;
        return (this->space == syncStage->space || this->space->isParentSpace(syncStage->space));
}

void FlowStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {

        LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;

        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
                                DependencyArc *arc = new DependencyArc(modifier, this, accessLog->getName());
                                arc->deriveSyncAndCommunicationRoots(hierarchy);

                                // Note that setting up outgoing arc on the source of the dependency happens inside
                                // the constructor of the DepedencyArc class. So here we consider the destination of 
				// the arc only.  
                                dataDependencies->addIncomingArcIfNotExists(arc);
                        }
                }
                if (accessLog->isModified()) {
                        modifierPanel->setLastModifierOfVar(this, accessLog->getName());
                }
        }
}

FlowStage *FlowStage::getNearestCommonAncestor(FlowStage *other) {
        FlowStage *first = this;
        FlowStage *second = other;
        while (first->index != second->index) {
                if (first->index > second->index) {
                        first = first->parent;
                } else if (second->index > first->index) {
                        second = second->parent;
                }
        }
        return first;
}

StageSyncReqs *FlowStage::getAllSyncRequirements() { return synchronizationReqs; }

StageSyncDependencies *FlowStage::getAllSyncDependencies() { return syncDependencies; }

void FlowStage::analyzeSynchronizationNeeds() {

        List<DependencyArc*> *outgoingArcList = dataDependencies->getOutgoingArcs();
        synchronizationReqs = new StageSyncReqs(this);
        for (int i = 0; i < outgoingArcList->NumElements(); i++) {
                DependencyArc *arc = outgoingArcList->Nth(i);
                const char *varName = arc->getVarName();
                FlowStage *destination = arc->getDestination();
                Space *destLps = destination->getSpace();

                // If the destination and current flow stage's LPSes are the same then two scenarios are there for us to 
		// consider: either the variable is replicated or it has overlapping partitions among adjacent LPUs. The 
		// former case is handled here. The latter is taken care of by the sync stage's overriding implementation; 
		// therefore we ignore it. 
                if (destLps == space) {
                        if (space->isReplicatedInCurrentSpace(varName)) {
                                ReplicationSync *replication = new ReplicationSync();
                                replication->setVariableName(varName);
                                replication->setDependentLps(destLps);
                                replication->setWaitingComputation(destination);
                                replication->setDependencyArc(arc);
                                synchronizationReqs->addVariableSyncReq(varName, replication, true);
                        } else {
				// An additional case is considered for composite stages when they encompasses nested 
				// stages executing in lower level LPSes than the former are executing in. We cannot be 
				// sure that there is a need for synchronization in this case. So we conservatively 
				// impose a synchronization requirement.  	
				CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(destination);
				if (compositeStage != NULL 
						&& compositeStage->hasExecutingCodeInDescendentLPSes()) {
					Space *lowestLps = compositeStage->getFurthestDescendentLpsWithinNestedFlow();	
					DownPropagationSync *downSync = new DownPropagationSync();
					downSync->setVariableName(varName);
					downSync->setDependentLps(lowestLps);
					downSync->setWaitingComputation(destination);
					downSync->setDependencyArc(arc);
					synchronizationReqs->addVariableSyncReq(varName, downSync, true);
				}
			}
                // If the destination and current flow stage's LPSes are not the same then there is definitely a 
		// synchronization need. The specific type of synchronization needed depends on the relative position of 
		// these two LPSes in the partition hierarchy.
                } else {
                        SyncRequirement *syncReq = NULL;
                        if (space->isParentSpace(destLps)) {
                                syncReq = new UpPropagationSync();
                        } else if (destLps->isParentSpace(space)) {
                                syncReq = new DownPropagationSync();
                        } else {
                                syncReq = new CrossPropagationSync();
                        }
                        syncReq->setVariableName(varName);
                        syncReq->setDependentLps(destLps);
                        syncReq->setWaitingComputation(destination);
                        syncReq->setDependencyArc(arc);
                        synchronizationReqs->addVariableSyncReq(varName, syncReq, true);
                }
        }
        synchronizationReqs->removeRedundencies();
}

void FlowStage::setReactivatorFlagsForSyncReqs() {

        List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
        List<SyncRequirement*> *filteredList = new List<SyncRequirement*>;
        if (syncList != NULL && syncList->NumElements() > 0) {
                for (int i = 0; i < syncList->NumElements(); i++) {
                        SyncRequirement *sync = syncList->Nth(i);
                        DependencyArc *arc = sync->getDependencyArc();

                        // if this is not the signal source then we can skip this arc as the signal source will
                        // take care of the reactivation flag processing
                        if (this != arc->getSignalSrc()) continue;

                        filteredList->Append(sync);
                }
        }
        // if the filtered list is not empty then there is some need for reactivation signaling
        if (filteredList->NumElements() > 0) {

                // reactivation signals should be received from PPUs of all LPSes that have computations dependent
                // on changes made by the current stage. So we need to partition the list into separate lists for 
		// individual LPSes
                Hashtable<List<SyncRequirement*>*> *syncMap = new Hashtable<List<SyncRequirement*>*>;
                for (int i = 0; i < filteredList->NumElements(); i++) {
                        SyncRequirement *sync = filteredList->Nth(i);
                        Space *dependentLps = sync->getDependentLps();
                        List<SyncRequirement*> *lpsSyncList = syncMap->Lookup(dependentLps->getName());
                        if (lpsSyncList == NULL) {
                                lpsSyncList = new List<SyncRequirement*>;
                        }
                        lpsSyncList->Append(sync);
                        syncMap->Enter(dependentLps->getName(), lpsSyncList, true);
                }

		// iterate over the sync list correspond to each LPS and select a reactivating sync for that LPS 
                Iterator<List<SyncRequirement*>*> iterator = syncMap->GetIterator();
                List<SyncRequirement*> *lpsSyncList;
                while ((lpsSyncList = iterator.GetNextValue()) != NULL) {

                        DependencyArc *closestPreArc = NULL;
                        DependencyArc *furthestSuccArc = NULL;

                        for (int i = 0; i < lpsSyncList->NumElements(); i++) {
                                SyncRequirement *sync = lpsSyncList->Nth(i);
                                DependencyArc *arc = sync->getDependencyArc();

                                FlowStage *syncStage = arc->getSignalSink();
                                int syncIndex = syncStage->getIndex();
                                if (syncIndex < this->index
                                                && (closestPreArc == NULL
                                                || syncIndex > closestPreArc->getSignalSink()->getIndex())) {
                                        closestPreArc = arc;
                                } else if (syncIndex >= this->index
                                                && (furthestSuccArc == NULL
                                                || syncIndex > furthestSuccArc->getSignalSink()->getIndex())) {
                                        furthestSuccArc = arc;
                                }
                        }

                        // The reactivator should be the last stage that will read changes made by this stage 
                        // before this stage can execute again. That will be the closest predecessor, if exists, 
                        // or the furthest successor.
                        if (closestPreArc != NULL) {
                                closestPreArc->setReactivator(true);
                        } else if (furthestSuccArc != NULL) {
                                furthestSuccArc->setReactivator(true);
                        } else {
                                std::cout << "This is strange!!! No reactivator is found for stage: ";
                                std::cout << this->getName() << std::endl;
                                std::exit(EXIT_FAILURE);
                        }
                }
        }
}

void FlowStage::printSyncRequirements(int indentLevel) {
        for (int i = 0; i < indentLevel; i++) std::cout << '\t';
        std::cout << "Stage: " << getName() << "\n";
        synchronizationReqs->print(indentLevel + 1);
}

List<const char*> *FlowStage::filterInArraysFromAccessMap(Hashtable<VariableAccess*> *accessMap) {
        if (accessMap == NULL) {
                accessMap = this->accessMap;
        }
        List<const char*> *arrayList = new List<const char*>;
        Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iterator.GetNextValue()) != NULL) {
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getLocalStructure(varName);
                if (structure == NULL) continue;
                ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
                if (array != NULL) {
                        arrayList->Append(varName);
                }
        }
        return arrayList;
}

bool FlowStage::isDependentStage(FlowStage *suspectedDependent) {
        if (synchronizationReqs == NULL) return false;
        return synchronizationReqs->isDependentStage(suspectedDependent);
}

List<const char*> *FlowStage::getVariablesNeedingCommunication(int segmentedPPS) {

        if (synchronizationReqs == NULL) return NULL;
        List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
        List<const char*> *varList = new List<const char*>;
        for (int i = 0; i < syncList->NumElements(); i++) {
                SyncRequirement *syncReq = syncList->Nth(i);
                bool commNeeded = syncReq->getCommunicationInfo(segmentedPPS)->isCommunicationRequired();
                const char *varName = syncReq->getVariableName();
                if (commNeeded && !string_utils::contains(varList, varName)) {
                        varList->Append(varName);
                }
        }
        return varList;
}

List<CommunicationCharacteristics*> *FlowStage::getCommCharacteristicsForSyncReqs(int segmentedPPS) {
        if (synchronizationReqs == NULL) return NULL;
        List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
        List<CommunicationCharacteristics*> *commCharList = new List<CommunicationCharacteristics*>;
        for (int i = 0; i < syncList->NumElements(); i++) {
                SyncRequirement *syncReq = syncList->Nth(i);
                commCharList->Append(syncReq->getCommunicationInfo(segmentedPPS));
        }
        return commCharList;
}
