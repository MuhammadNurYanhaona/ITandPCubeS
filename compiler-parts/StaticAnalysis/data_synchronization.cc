#include "ast.h"
#include "ast_expr.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "task_space.h"
#include "data_access.h"
#include "data_flow.h"
#include "hashtable.h"
#include "list.h"

//----------------------------------------------- Space Entry Checkpoint --------------------------------------------------/

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

//----------------------------------------------- Sync Stage Generator --------------------------------------------------/
        
bool SyncStageGenerator::doesTransitionNeedSynchronization(Space *previousSpace, Space *nextSpace) {
	return false;
}

SyncStage *SyncStageGenerator::generateEntrySyncStage(Space *space) {
	if (space->isSubpartitionSpace()) {
		return new SyncStage(space, Load_And_Configure, Entrance);
	} else if (space->isDynamic()) {
		return new SyncStage(space, Load, Entrance);
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
	SyncStage *syncStage  = new SyncStage(space, Ghost_Region_Update, Reappearance);
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
		SyncStage *syncStage = new SyncStage(space, Restore, Return);
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

//---------------------------------- Exit Space to Data Structure Mappings -----------------------------------------/

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
	SyncStage *syncStage = new SyncStage(ancestorSpace, Restore, Exit);
	for (int i = 0; i < accessList->NumElements(); i++) {
		syncStage->addAccessInfo(accessList->Nth(i));	
	}
	return syncStage;
}
