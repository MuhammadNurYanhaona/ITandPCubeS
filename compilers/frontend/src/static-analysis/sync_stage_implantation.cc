#include "sync_stage_implantation.h"
#include "sync_stat.h"
#include "data_dependency.h"
#include "../common/constant.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"
#include "../semantics/data_access.h"
#include "../semantics/computation_flow.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

//------------------------------------------------------- Sync Stage -------------------------------------------------------/

SyncStage::SyncStage(Space *space, SyncMode mode, SyncStageType type) : FlowStage(space) {
	this->mode = mode;
	this->type = type;
	if (mode == Load || mode == Load_And_Configure) {
                this->name = "\"Data Loader Sync\"";
        } else if (mode == Ghost_Region_Update) {
                this->name = "\"Ghost Region Updater Sync\"";
        } else {
                this->name = "\"Data Restorer Sync\"";
        }
	this->prevDataModifiers = new Hashtable<FlowStage*>;
}

void SyncStage::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Sync Stage: ";
	if (name != NULL) {
		std::cout << name << " ";
	}
        std::cout << "(Space " << space->getName() << ")\n"; 
	if (name == NULL) {
		std::cout << indent.str() << "Mode: ";
		if (mode == Load) {
			std::cout << "Load";
		} else if (mode == Load_And_Configure) {
			std::cout << "Load and Configure";
		} else if (mode == Ghost_Region_Update) {
			std::cout << "Update Ghost Region";
		} else {
			std::cout << "Restore";
		}
		std::cout << "\n"; 
	}
	Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess* accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                accessLog->printAccessDetail(indentLevel + 1);
        }
	dataDependencies->print(indentLevel + 1);
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

void SyncStage::addAccessInfo(VariableAccess *accessLog) {
        const char *varName = accessLog->getName();
        if (accessMap->Lookup(varName) != NULL) accessMap->Lookup(varName)->mergeAccessInfo(accessLog);
        else accessMap->Enter(varName, accessLog, true);
}

void SyncStage::performDependencyAnalysis(PartitionHierarchy *hierarchy) {

        LastModifierPanel *modifierPanel = LastModifierPanel::getPanel();
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;

        // just save a reference to the last modifier of any variable this sync stage is interested in
        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (accessLog->isRead() || accessLog->isModified()) {
                        FlowStage *modifier = modifierPanel->getLastModifierOfVar(accessLog->getName());
                        if (modifier != NULL && modifier != this && isDataModifierRelevant(modifier)) {
                                prevDataModifiers->Enter(accessLog->getName(), modifier);
                        }
                }
        }

        // then call the superclass's dependency analysis function
        FlowStage::performDependencyAnalysis(hierarchy);
}

void SyncStage::analyzeSynchronizationNeeds() {

        // the general logic for setting up synchronization requirements works for all sync stages except the
        // ghost region sync stages
        if (mode != Ghost_Region_Update) {
                FlowStage::analyzeSynchronizationNeeds();
        } else {
                List<DependencyArc*> *outgoingArcList = dataDependencies->getOutgoingArcs();
                synchronizationReqs = new StageSyncReqs(this);
                for (int i = 0; i < outgoingArcList->NumElements(); i++) {
                         
                        DependencyArc *arc = outgoingArcList->Nth(i);
                        const char *varName = arc->getVarName();
                        FlowStage *destination = arc->getDestination();

                        // if the destination is another ghost region sync stage then we ignore this dependency
                        // as the communication model ensures that such dependencies resolves automatically
                        SyncStage *destSync = dynamic_cast<SyncStage*>(destination);
                        if (destSync != NULL && destSync->mode == Ghost_Region_Update) continue;

                        Space *destLps = destination->getSpace();
                        DataStructure *structure = space->getLocalStructure(varName);
                        ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
                        if (array == NULL) continue;
                        if (!array->hasOverlappingsAmongPartitions()) continue;
                        List<int> *overlappingDims = array->getOverlappingPartitionDims();
                        GhostRegionSync *ghostRegion = new GhostRegionSync();
                        ghostRegion->setVariableName(varName);
                        ghostRegion->setDependentLps(destLps);
                        ghostRegion->setWaitingComputation(destination);
                        ghostRegion->setDependencyArc(arc);
                        ghostRegion->setOverlappingDirections(overlappingDims);
                        synchronizationReqs->addVariableSyncReq(varName, ghostRegion, true);
                }
        }

        // indicate that the synchronization requirements registered by the stage do not need any execution 
        // counter to be activated -- data resolution must be done here regardless of the execution of any stage
        if (synchronizationReqs != NULL) {
                List<SyncRequirement*> *syncList = synchronizationReqs->getAllSyncRequirements();
                for (int i = 0; i < syncList->NumElements(); i++) {
                        syncList->Nth(i)->setCounterRequirement(false);
                }       
        }
}

FlowStage *SyncStage::getUltimateModifier(const char *varName) {
        FlowStage *lastModifier = prevDataModifiers->Lookup(varName);
        if (lastModifier == NULL) return NULL;
        SyncStage *previousSync = dynamic_cast<SyncStage*>(lastModifier);
        if (previousSync == NULL) return lastModifier;
        return previousSync->getUltimateModifier(varName);
}

//-------------------------------------------------- Space Entry Checkpoint ------------------------------------------------/


SpaceEntryCheckpoint::SpaceEntryCheckpoint(Space *space , int entryStageIndex) {
        this->space = space;
        this->entryStageIndex = entryStageIndex;
        this->entrySyncStage = NULL;
}

Hashtable<SpaceEntryCheckpoint*> *SpaceEntryCheckpoint::checkpointList = new Hashtable<SpaceEntryCheckpoint*>;

SpaceEntryCheckpoint *SpaceEntryCheckpoint::getCheckpoint(Space *space) {
        return SpaceEntryCheckpoint::checkpointList->Lookup(space->getName());
}

void SpaceEntryCheckpoint::removeACheckpoint(Space *space) {
        const char *spaceName = space->getName();
        SpaceEntryCheckpoint *checkpoint = SpaceEntryCheckpoint::checkpointList->Lookup(spaceName);
        if (checkpoint != NULL) {
                SpaceEntryCheckpoint::checkpointList->Remove(spaceName, checkpoint);
        }
}

SpaceEntryCheckpoint *SpaceEntryCheckpoint::addACheckpointIfApplicable(Space *space, int stageIndex) {
        const char *spaceName = space->getName();
        SpaceEntryCheckpoint *checkpoint = SpaceEntryCheckpoint::checkpointList->Lookup(spaceName);
        if (checkpoint == NULL) {
                checkpoint = new SpaceEntryCheckpoint(space, stageIndex);
                SpaceEntryCheckpoint::checkpointList->Enter(spaceName, checkpoint, true);
                return checkpoint;
        }
        return NULL;
}

//--------------------------------------------------- Sync Stage Generator -------------------------------------------------/

bool SyncStageGenerator::doesTransitionNeedSynchronization(Space *previousSpace, Space *nextSpace) {
        return false;
}

SyncStage *SyncStageGenerator::generateEntrySyncStage(Space *space) {
        if (space->isSubpartitionSpace()) {
                return new SyncStage(space, Load_And_Configure, Entrance_Sync);
        } else if (space->isDynamic()) {
                return new SyncStage(space, Load, Entrance_Sync);
        }
        return NULL;
}

void SyncStageGenerator::populateAccessMapOfEntrySyncStage(SyncStage *stage, Hashtable<VariableAccess*> *accessLogs) {
        List<VariableAccess*> *accessList = SyncStageGenerator::generateListFromLogs(accessLogs);
        if (accessList->NumElements() == 0) return;
        List<const char*> *spaceLocalDataStructures = stage->getSpace()->getLocalDataStructureNames();
        List<VariableAccess*> *filteredList = SyncStageGenerator::filterAccessList(accessList, spaceLocalDataStructures);
        if (filteredList->NumElements() == 0) return;
        stage->populateAccessMap(filteredList, true, false);
}

SyncStage *SyncStageGenerator::generateReappearanceSyncStage(Space *space, Hashtable<VariableAccess*> *accessLogs) {
        List<VariableAccess*> *accessList = SyncStageGenerator::generateListFromLogs(accessLogs);
        if (accessList->NumElements() == 0) return NULL;
        List<const char*> *overlappedStructures = space->getLocalDataStructuresWithOverlappedPartitions();
        if (overlappedStructures->NumElements() == 0) return NULL;
        List<VariableAccess*> *filteredList = SyncStageGenerator::filterAccessList(accessList, overlappedStructures);
        if (filteredList->NumElements() == 0) return NULL;
        SyncStage *syncStage  = new SyncStage(space, Ghost_Region_Update, Reappearance_Sync);
        int syncVarCount = syncStage->populateAccessMap(filteredList, false, true);
        return (syncVarCount > 0) ? syncStage : NULL;
}

SyncStage *SyncStageGenerator::generateReturnSyncStage(Space *space, Hashtable<VariableAccess*> *accessLogs) {
        if (space->isSubpartitionSpace() || space->isDynamic()) {
                List<VariableAccess*> *accessList = SyncStageGenerator::generateListFromLogs(accessLogs);
                if (accessList->NumElements() == 0) return NULL;
                List<const char*> *spaceLocalDataStructures = space->getLocalDataStructureNames();
                List<VariableAccess*> *filteredList = SyncStageGenerator::filterAccessList(accessList, spaceLocalDataStructures);
                if (filteredList->NumElements() == 0) return NULL;
                SyncStage *syncStage = new SyncStage(space, Restore, Return_Sync);
                int syncVarCount = syncStage->populateAccessMap(filteredList, false, true);
                return (syncVarCount > 0) ? syncStage : NULL;
        }
        return NULL;
}

List<SyncStage*> *SyncStageGenerator::generateExitSyncStages(Space *space, Hashtable<VariableAccess*> *accessLogs) {

        List<SyncStage*> *synStageList = new List<SyncStage*>;
        List<const char*> *concernedIfUpdated = NULL;
        if (space->isDynamic()) concernedIfUpdated = space->getLocalDataStructureNames();
        else concernedIfUpdated = space->getNonStorableDataStructures();
        if (concernedIfUpdated == NULL || concernedIfUpdated->NumElements() == 0) return synStageList;

        List<VariableAccess*> *accessList = SyncStageGenerator::generateListFromLogs(accessLogs);
        List<VariableAccess*> *filteredList = SyncStageGenerator::filterAccessList(accessList, concernedIfUpdated);
        List<const char*> *finalList = new List<const char*>;
        for (int i = 0; i < filteredList->NumElements(); i++) {
                VariableAccess *accessLog = filteredList->Nth(i);
                if (accessLog->isContentAccessed()
                        && (accessLog->getContentAccessFlags()->isWritten()
                                || accessLog->getContentAccessFlags()->isRedirected())) {
                        finalList->Append(accessLog->getName());
                }
        }
        if (finalList->NumElements() == 0) return synStageList;

        List<const char*> *alternateList = new List<const char*>;
        Space *ancestorSpace = space;
        while ((ancestorSpace = ancestorSpace->getParent()) != NULL) {
                ExitSpaceToDataStructureMappings *mappings = new ExitSpaceToDataStructureMappings(ancestorSpace);
                for (int i = 0; i < finalList->NumElements(); i++) {
                        const char *varName = finalList->Nth(i);
                        bool isInSpace = (ancestorSpace->getLocalStructure(varName) != NULL);
                        if (isInSpace) {
                                mappings->generateAccessInfo(varName);
                        } else {
                                alternateList->Append(varName);
                        }
                }
                if (!mappings->isAccessListEmpty()) synStageList->Append(mappings->generateSyncStage());
                if (alternateList->NumElements() == 0) break;
                finalList = alternateList;
                alternateList = new List<const char*>;
        }
        return synStageList;
}

List<VariableAccess*> *SyncStageGenerator::generateListFromLogs(Hashtable<VariableAccess*> *accessLogs) {
        List<VariableAccess*> *accessList = new List<VariableAccess*>;
        Iterator<VariableAccess*> iter = accessLogs->GetIterator();
        VariableAccess *accessLog = NULL;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                accessList->Append(accessLog);
        }
        return accessList;
}

List<VariableAccess*> *SyncStageGenerator::filterAccessList(List<VariableAccess*> *accessList,
                        List<const char*> *includeIfExistsList) {
        List<VariableAccess*> *filteredList = new List<VariableAccess*>;
        for (int i = 0; i < accessList->NumElements(); i++) {
                VariableAccess *accessLog = accessList->Nth(i);
                const char *varName = accessLog->getName();
                for (int j = 0; j < includeIfExistsList->NumElements(); j++) {
                        if (strcmp(varName, includeIfExistsList->Nth(j)) == 0) {
                                filteredList->Append(accessLog);
                                break;
                        }
                }
        }
        return filteredList;
}

//------------------------------------------- Exit Space to Data Structure Mappings ----------------------------------------/

ExitSpaceToDataStructureMappings::ExitSpaceToDataStructureMappings(Space *ancestorSpace) {
        this->ancestorSpace = ancestorSpace;
        this->accessList = new List<VariableAccess*>;
}

void ExitSpaceToDataStructureMappings::generateAccessInfo(const char *varName) {
        VariableAccess *accessLog = new VariableAccess(varName);
        accessLog->markContentAccess();
        accessLog->getContentAccessFlags()->flagAsWritten();
        accessLog->getContentAccessFlags()->flagAsRead();
        accessList->Append(accessLog);
}

SyncStage *ExitSpaceToDataStructureMappings::generateSyncStage() {
        SyncStage *syncStage = new SyncStage(ancestorSpace, Restore, Exit_Sync);
        for (int i = 0; i < accessList->NumElements(); i++) {
                syncStage->addAccessInfo(accessList->Nth(i));
        }
        return syncStage;
}

