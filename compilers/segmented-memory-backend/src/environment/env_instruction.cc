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
		versionAttr->setPartContainerTree(allocation->getContainerTree());
		versionAttr->computeSegmentFold();
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

		versionManager->addFreshVersionKey(versionKey);
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

bool TaskInitEnvInstruction::isFresh(TaskItem *envItem) {
	
	TaskEnvironment *taskEnv = envItem->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = envItem->getEnvLinkKey();
	const char *itemName = envKey->getVarName();

	ListReferenceKey *patternKey = ListReferenceKey::initiatePatternKey();
	patternKey->setTaskEnvId(envId);
	patternKey->setVarName(itemName);

        ObjectVersionManager *versionManager = progEnv->getVersionManager(envKey->getSourceKey());
	if (versionManager == NULL) {
		std::cout << "No data parts version manager is found in the program environment for ";
		std::cout << itemName << " used in " << taskEnv->name << "\n";
		std::exit(EXIT_FAILURE);
	}

	return versionManager->foundMatchingFreshVersion(patternKey);
}

void TaskInitEnvInstruction::cloneDataFromPartsList(const char *itemKey, 
		ListReferenceKey *sourcePartsListKey, 
		PartsList *destination) {
	
	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
        ObjectVersionManager *versionManager = progEnv->getVersionManager(itemKey);
	
	PartsListReference *sourceReference = versionManager->getVersion(sourcePartsListKey->generateKey());
	PartsList *source = sourceReference->getPartsList();
	
	List<DataPart*> *sourceParts = source->getDataParts();
	List<DataPart*> *destDataParts = destination->getDataParts();

	// note that the assumption when this function has been called is that the source and destination part lists have
	// the same configuration but the destination is just lacking the up-to-date contents of the underlying parts
	if (destDataParts == NULL || destDataParts->NumElements() == 0) return;
	for (int i = 0; i < destDataParts->NumElements(); i++) {
		DataPart *destPart = destDataParts->Nth(i);
		DataPart *sourcePart = sourceParts->Nth(i);
		destPart->clone(sourcePart);
	}
}

//--------------------------------------------------- StaleRefreshInstruction -----------------------------------------------------

void StaleRefreshInstruction::setupPartsList() {

	// create a new list for dimension information as opposed to get the list from the item as the latter can be
	// updated any time 
	int dimensionality = itemToUpdate->getDimensionality();
	List<Dimension> *rootDimensions = new List<Dimension>;
	for (int i = 0; i < dimensionality; i++) {
		rootDimensions->Append(itemToUpdate->getDimension(i));
	}

	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *itemKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = itemKey->getVarName();
	const char *itemSourceKey = itemKey->getSourceKey();
        ObjectVersionManager *versionManager = progEnv->getVersionManager(itemSourceKey);

	// If the parts lists for the item reference are still fresh in the environment then we only need to copy the
	// contents of the data parts from environmental parts lists to the parts lists manipulated by the task executor.
	// This is needed because data parts are not kept with the task environment after a task ends execution; rather
	// the program environment maintains them. This strategy is adopted because a single task can be invoked with
	// different partition parameters at different times. Therefore despite the parts list being fresh, there might
	// be a need for reorganizing the data parts from their previous state.    
	if (isFresh(itemToUpdate)) {
		Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
		LpsAllocation *allocation = NULL;
		Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
		while ((allocation = iterator.GetNextValue()) != NULL) {
			DataPartitionConfig *allocationConfig = allocation->getPartitionConfig();
			ListReferenceKey *versionKey 
					= allocation->generatePartsListReferenceKey(envId, itemName);
			PartsListReference *envReference = versionManager->getVersion(versionKey->generateKey());
			DataPartitionConfig *envRefConfig = envReference->getAttributes()->getPartitionConfig();
			
			// if the two data partition configurations are the same then the data parts are the same and
			// there is no need for data reorganization
			if (allocationConfig->isEquivalent(envRefConfig)) {
				cloneDataFromPartsList(itemSourceKey, versionKey, allocation->getPartsList());
				continue;
			}
			// otherwise, there needs to be a reorganization of data parts which will most likely lead to
			// cross-segment communications

			// first we need to allocate the data parts in the LPS allocation list
			allocation->allocatePartsList();
			
			// then we need to generate a communicator to transfer data from the source to the allocation
			
			// then we need to remove the source parts list reference from the version manager
			versionManager->removeVersion(versionKey);

			// then we need to add the parts list of the LPS allocation as a new version reference in the 
			// program environment
			ListReferenceAttributes *attr 
					= new ListReferenceAttributes(allocationConfig, rootDimensions);
			attr->setPartContainerTree(allocation->getContainerTree());
			attr->computeSegmentFold();
			PartsList *partsList = allocation->getPartsList();
			PartsListReference *versionRef = new PartsListReference(attr, versionKey, partsList);
			versionManager->addNewVersion(versionRef);
		}
		return;	
	}

	// If the parts lists for the current task item are stale then we need to transfer data from parts list created
	// by other tasks that are fresh on LPS allocation by LPS allocation basis. The logic for identifying the best 
	// fresh parts list version to fill in data from is to look for a version whose partition configuration matches
	// the current item's configuration. If such a version is found then data parts content can be just copied from 
	// it to the current allocation and that version's reference count needs to be increased only.
	//
	// Otherwise, the first fresh version is retrieved to maintain uniformity across segments and necessary steps 
	// are taken for data transfers from that version to the LPS allocation's parts list after memory allocation has
	// been done for the latter's data parts.
	List<PartsListReference*> *freshVersionList = versionManager->getFreshVersions();
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	LpsAllocation *allocation = NULL;
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	while ((allocation = iterator.GetNextValue()) != NULL) {
		DataPartitionConfig *allocationConfig = allocation->getPartitionConfig();
		PartsListReference *sourceVersion = freshVersionList->Nth(0);
		bool matchingConfigFound = false;
		for (int i = 0; i < freshVersionList->NumElements(); i++) {
			PartsListReference *reference = freshVersionList->Nth(i);
			DataPartitionConfig *envRefConfig = reference->getAttributes()->getPartitionConfig();
			if (allocationConfig->isEquivalent(envRefConfig)) {
				matchingConfigFound = true;
				sourceVersion = reference;
				break;
			}
		}
		if (matchingConfigFound) {
			// copy data parts' contents from the source reference to the allocation's parts list
			ListReferenceKey *versionKey = sourceVersion->getKey();
			cloneDataFromPartsList(itemSourceKey, versionKey, allocation->getPartsList());
			// increase the reference count of the parts list in the environment
			sourceVersion->getPartsList()->getAttributes()->increaseReferenceCount();
		} {
			// allocate memory for the LPS allocation's parts list
			allocation->allocatePartsList();

			// create a communicator to transfer data from the environment reference to the LPS allocation

			// register a new version for the allocation's parts list in the environment
			ListReferenceKey *versionKey = allocation->generatePartsListReferenceKey(envId, itemName);
			ListReferenceAttributes *attr 
					= new ListReferenceAttributes(allocationConfig, rootDimensions);
			attr->setPartContainerTree(allocation->getContainerTree());
			attr->computeSegmentFold();
			PartsList *partsList = allocation->getPartsList();
			PartsListReference *versionRef = new PartsListReference(attr, versionKey, partsList);
			versionManager->addNewVersion(versionRef);
		}
	}
}

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

void DataTransferInstruction::setupPartsList() {
	
	// create a new list for dimension information as opposed to get the list from the item as the latter can be
	// updated any time 
	int dimensionality = itemToUpdate->getDimensionality();
	List<Dimension> *rootDimensions = new List<Dimension>;
	for (int i = 0; i < dimensionality; i++) {
		rootDimensions->Append(itemToUpdate->getDimension(i));
	}

	TaskEnvironment *taskEnv = itemToUpdate->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = itemToUpdate->getEnvLinkKey();
	const char *itemName = envKey->getVarName();
	
	ArrayTransferConfig *collapsedConfig = transferConfig->getCollapsed();
	const char *sourceProperty = collapsedConfig->getPropertyName();
	
	if (sourceProperty == NULL) {
		std::cout << "Currently we do not support a non environmental data structure to directly serve ";
		std::cout << "as a data source for a task's environment\n";
		std::cout << "So cannot process instruction for " << itemName << " in " << taskEnv->name << "\n";
		std::exit(EXIT_FAILURE);
	}

	void *source = collapsedConfig->getSource();
	TaskEnvironment *sourceEnv = (TaskEnvironment*) source;
	TaskItem *sourceItem = sourceEnv->getItem(sourceProperty);

	// setup the data item key for the to-be-updated task item to be the same as that of the source item as both are
	// referring to the same data structure in the environment
	const char *dataSourceKey = sourceItem->getEnvLinkKey()->getSourceKey();
	itemToUpdate->getEnvLinkKey()->setSourceKey(dataSourceKey);

	// Iterate over the LPS allocations of the to be updated task item to be filled with data-parts associated with 
	// the source task item. First, existing parts-list reference in the program environment for the LPS allocation
	// should be removed as it is going to be updated. Then a search is made over the LPS allocations of the source
	// task item to identify if any extant parts list associated with the item can directly be copied to the current
	// LPS allocation. If no such parts list is found then the first parts list of the source task item is selected
	// to maintain uniformity of logic across segments to populate the current allocation through data transfer 
	// that might involve cross segment communication.  
	Hashtable<LpsAllocation*> *allocationMap = itemToUpdate->getAllAllocations();
	LpsAllocation *allocation = NULL;
	Iterator<LpsAllocation*> iterator = allocationMap->GetIterator();
	while ((allocation = iterator.GetNextValue()) != NULL) {
		DataPartitionConfig *destConfig = allocation->getPartitionConfig();
		Iterator<LpsAllocation*> sourceIterator = sourceItem->getAllAllocations()->GetIterator();
		LpsAllocation *candidateAllocation = NULL;
		LpsAllocation *selectedAllocation = NULL;
		bool matchingFound = false;
		while ((candidateAllocation = sourceIterator.GetNextValue()) != NULL) {
			if (selectedAllocation == NULL) {
				selectedAllocation = candidateAllocation;
			}
			DataPartitionConfig *candidateConfig = candidateAllocation->getPartitionConfig();
			if (destConfig->isEquivalent(candidateConfig)) {
				matchingFound = true;
				selectedAllocation = candidateAllocation;
				break;
			}
		}
		ListReferenceKey *referenceKey = selectedAllocation->generatePartsListReferenceKey(
				sourceEnv->getEnvId(), sourceProperty);
		ObjectVersionManager *versionManager = progEnv->getVersionManager(dataSourceKey);
		PartsListReference *envReference = versionManager->getVersion(referenceKey->generateKey());		

		// in case of a direct copy of parts list from the environment to the current allocation, the reference
		// count of the parts list should be increased
		if (matchingFound) {
			cloneDataFromPartsList(dataSourceKey, referenceKey, allocation->getPartsList());
			envReference->getPartsList()->getAttributes()->increaseReferenceCount();
		} else {
			// allocate memories for the parts in the LPS allocation
			allocation->allocatePartsList();
	
			// generate a communicator to transfer data from the source version reference in the environment
			// to the LPS allocation

			// generate a new version reference for the LPS allocation and add that to the version manager 
			ListReferenceKey *versionKey = allocation->generatePartsListReferenceKey(envId, itemName);
			ListReferenceAttributes *attr = new ListReferenceAttributes(destConfig, rootDimensions);
			attr->setPartContainerTree(allocation->getContainerTree());
			attr->computeSegmentFold();
			PartsList *partsList = allocation->getPartsList();
			PartsListReference *versionRef = new PartsListReference(attr, versionKey, partsList);
			versionManager->addNewVersion(versionRef);
		}

		// in either case, verify if the selected source parts list version is stale or fresh; if it is stale
		// then identify a fresh list and do data transfer from the selected fresh version to the parts list of 
		// the current allocation
		if (!envReference->getPartsList()->getAttributes()->isFresh()) {
			PartsListReference *freshVersionRef = versionManager->getFirstFreshVersion();

			// create communicator to transfer data from the fresh parts list version to the data parts of
			// the current allocation			
		} 	
	}
}

/*---------------------------------------------------------------------------------------------------------------------------------
                            Environment Instructions to be Processed At Task Completion or Program End
---------------------------------------------------------------------------------------------------------------------------------*/

//--------------------------------------------------- ChangeNotifyInstruction -----------------------------------------------------

void ChangeNotifyInstruction::updateProgramEnv() {

	TaskEnvironment *taskEnv = envItem->getEnvironment();
	int envId = taskEnv->getEnvId();
	ProgramEnvironment *progEnv = taskEnv->getProgramEnvironment();
	EnvironmentLinkKey *envKey = envItem->getEnvLinkKey();
	const char *itemName = envKey->getVarName();

	ListReferenceKey *patternKey = ListReferenceKey::initiatePatternKey();
	patternKey->setTaskEnvId(envId);
	patternKey->setVarName(itemName);

        ObjectVersionManager *versionManager = progEnv->getVersionManager(envKey->getSourceKey());
	versionManager->markNonMatchingVersionsStale(patternKey);		
}

