#include "env_instruction.h"
#include "environment.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../input-output/stream.h"
#include "../runtime/structure.h"
#include "../runtime/array_transfer.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

/*---------------------------------------------------------------------------------------------------------------------------------
				    Environment Instructions to be Processed At Task Initialization
---------------------------------------------------------------------------------------------------------------------------------*/

//---------------------------------------------------- TaskInitEnvInstruction -----------------------------------------------------

void TaskInitEnvInstruction::removeOldPartsListReferences() {
	
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();

	// if the task item has a valid source key assigned to it then it is not the first invocation of the task using the 
	// current environment, and we need to let go of the older parts list reference the task has for the item 
	const char *sourceKey = envKey->getSourceKey();	
	if (sourceKey != NULL) {
		
		// delete parts list references for different LPS allocations the task have for the item
		ObjectVersionManager *versionMg = progEnv->getVersionManager(sourceKey);
		Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
		Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
		LpsAllocation *allocation = NULL;
		while ((allocation = iterator.GetNextValue()) != NULL) {
			ListReferenceKey *versionKey = allocation->generatePartsListReferenceKey(envId, itemName);
			versionMg->removeVersion(versionKey);
		}

		// if current task's references to the item were the last then the item should be garbage collected
		progEnv->cleanupPossiblyEmptyVersionManager(sourceKey);
	}
}

void TaskInitEnvInstruction::allocatePartsLists() {
	
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	LpsAllocation *allocation = NULL;
	
	while ((allocation = iterator.GetNextValue()) != NULL) {
		PartsList *partsList = allocation->getPartsList();
		List<DataPart*> *dataParts = partsList->getDataParts();
		if (dataParts == NULL) return;

		for (int i = 0; i < dataParts->NumElements(); i++) {
			dataParts->Nth(i)->allocate();	
		}
	}
}

void TaskInitEnvInstruction::assignDataSourceKeyForItem() {
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int taskId = taskEnv->getTaskId();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	envKey->flagAsDataSource(taskId);
}

void TaskInitEnvInstruction::initiateVersionManagement() {
	
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int envId = taskEnv->getEnvId();
	int taskId = taskEnv->getTaskId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();

	// create a new list for dimension information as opposed to get the list from the item as the latter can be
	// updated any time 
	int dimensionality = itemToUpdate->getDimensionality();
	List<Dimension> *rootDimensions = new List<Dimension>;
	for (int i = 0; i < dimensionality; i++) {
		rootDimensions->Append(itemToUpdate->getDimension(i));
	}

	// version manager is initiallly NULL as the construction requires that there is at least one valid parts list
	// reference in it
	ObjectVersionManager *versionManager = NULL;
	ObjectIdentifier *itemId = envKey->generateObjectIdentifier(taskId);

	// create parts list references separately for individual LPS allocations and store them within the manager
	bool managerCreated = false;
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	LpsAllocation *allocation = NULL;
	while ((allocation = iterator.GetNextValue()) != NULL) {
		
		// create a parts-list version reference for the allocation
		ListReferenceKey *versionKey = allocation->generatePartsListReferenceKey(envId, itemName);
		DataPartitionConfig *partConfig = allocation->getPartitionConfig();
		ListReferenceAttributes *versionAttr = new ListReferenceAttributes(partConfig, rootDimensions);
		PartsList *partsList = allocation->getPartsList();
		PartsListReference *versionRef = new PartsListReference(versionAttr, versionKey, partsList);
		
		// for the first version reference, create a manager object by directly inserting the reference in the
		// program environment	
		if (!managerCreated) {
			progEnv->addNewDataItem(itemId, versionRef);
			versionManager = progEnv->getVersionManager(itemId->generateKey());
			managerCreated = true;
		// otherwise, just add the version to the manager
		} else {
			versionManager->addNewVersion(versionRef);
		}
	}
}

void TaskInitEnvInstruction::recordFreshPartsListVersions() {
	
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();
	ObjectVersionManager *versionManager = progEnv->getVersionManager(envKey->getSourceKey());
	
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	LpsAllocation *allocation = NULL;
	while ((allocation = iterator.GetNextValue()) != NULL) {
		allocation->getPartsList()->getAttributes()->flagFresh();
		ListReferenceKey *versionKey = allocation->generatePartsListReferenceKey(envId, itemName);
		versionManager->addFreshVersionKey(versionKey);
	}
	
}

//--------------------------------------------------- StaleRefreshInstruction -----------------------------------------------------


//---------------------------------------------------- CreateFreshInstruction -----------------------------------------------------

void CreateFreshInstruction::setupPartsList() {
	allocatePartsLists();
	assignDataSourceKeyForItem();	
}

//--------------------------------------------------- ReadFromFileInstruction -----------------------------------------------------

void ReadFromFileInstruction::setupDimensions() {

	TaskEnvironment *environment = itemToUpdate->getEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();
	TypedInputStream<char> *stream = new TypedInputStream<char>(fileName);
	List<Dimension*> *dimensionList = stream->getDimensionList();
	for (int i = 0; i < dimensionList->NumElements(); i++) {
		Dimension *dimension = dimensionList->Nth(i);
		itemToUpdate->setDimension(i, *dimension);
	}
	delete stream;		
}

void ReadFromFileInstruction::setupPartsList() {
	
	allocatePartsLists();
	
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	LpsAllocation *allocation = NULL;

	while ((allocation = iterator.GetNextValue()) != NULL) {
		const char *lpsId = allocation->getLpsId();
		PartsList *partsList = allocation->getPartsList();
		List<DataPart*> *dataParts = partsList->getDataParts();
		if (dataParts == NULL) return;

		PartReader *partReader = taskEnv->getPartReader(itemName, lpsId);		
		partReader->setFileName(fileName);
		partReader->processParts();
	}	
	
	assignDataSourceKeyForItem();	
}

//--------------------------------------------------- DataTransferInstruction -----------------------------------------------------

void DataTransferInstruction::setupDimensions() {
	
	int dimensionality = itemToUpdate->getDimensionality();
	for (int i = 0; i < dimensionality; i++) {
		Range dimRange = transferConfig->getNearestDimTransferRange(i);
		Dimension dimension = Dimension();
		dimension.range = dimRange;
		dimension.setLength();
		itemToUpdate->setDimension(i, dimension);
	}
}

