#include "environment.h"
#include "env_instruction.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/interval.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_tracking.h"
#include "../communication/part_folding.h"
#include "../communication/part_config.h"
#include "../runtime/structure.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

using namespace std;

//---------------------------------------------------------- Segment Data Content -------------------------------------------------------/

SegmentDataContent::SegmentDataContent(int segmentId, char *stringFoldDesc) {
	this->segmentId = segmentId;
	this->stringFoldDesc = stringFoldDesc;
}

SegmentDataContent::~SegmentDataContent() {
	delete[] stringFoldDesc;
}

List<MultidimensionalIntervalSeq*> *SegmentDataContent::generateFold() {
	return MultidimensionalIntervalSeq::constructSetFromString(stringFoldDesc);
}

//--------------------------------------------------------- Parts List Attributes -------------------------------------------------------/

PartsListAttributes::PartsListAttributes() {
	referenceCount = 1;
	fresh = true;
	dirty = false;
	segmentMappingKnown = false;
	segmentsContents = NULL;
}

PartsListAttributes::~PartsListAttributes() {
	if (segmentsContents != NULL) {
		while (segmentsContents->NumElements() > 0) {
			SegmentDataContent *segmentContent = segmentsContents->Nth(0);
			segmentsContents->RemoveAt(0);
			delete segmentContent;
		}
		delete segmentsContents;
		segmentsContents = NULL;
	}
}

void PartsListAttributes::setSegmentsContents(List<SegmentDataContent*> *segmentsContents) {
	this->segmentsContents = segmentsContents;
	segmentMappingKnown = true;
}

//--------------------------------------------------------------- Parts List ------------------------------------------------------------/

PartsList::PartsList(List<DataPart*> *parts) {
	this->attributes = new PartsListAttributes();
	this->parts = parts;
}

PartsList::PartsList(PartsListAttributes *attributes, List<DataPart*> *parts) {
	this->attributes = attributes;
	this->parts = parts;
}
        
PartsList::~PartsList() {
	delete attributes;
	while (parts->NumElements() > 0) {
		DataPart *part = parts->Nth(0);
		parts->RemoveAt(0);
		delete part;
	}
	delete parts;
}

//-------------------------------------------------------- List Reference Attributes ----------------------------------------------------/

ListReferenceAttributes::ListReferenceAttributes(DataPartitionConfig *partitionConfig, List<Dimension> *rootDimensions) {
	this->partitionConfig = partitionConfig;
	this->rootDimensions = rootDimensions;
	this->partContainerTree = NULL;
	this->segmentFold = NULL;
}
        
ListReferenceAttributes::~ListReferenceAttributes() {
	if (partContainerTree != NULL) {
		delete partContainerTree;
	}
	if (segmentFold != NULL) {
		while (segmentFold->NumElements() > 0) {
			MultidimensionalIntervalSeq *seq = segmentFold->Nth(0);
			segmentFold->RemoveAt(0);
			delete seq;
		}
		delete segmentFold;
	}
}

void ListReferenceAttributes::computeSegmentFold(std::ofstream *logFile) {

	segmentFold = ListReferenceAttributes::computeSegmentFold(partitionConfig, 
			partContainerTree, logFile);
}

void ListReferenceAttributes::printSegmentFold(std::ofstream &stream) {
	std::ostringstream desc;
	desc << "Segment Fold:\n";
	if (segmentFold != NULL) {
		for (int i = 0; i < segmentFold->NumElements(); i++) {
			MultidimensionalIntervalSeq *seq = segmentFold->Nth(i);
			seq->draw(1, desc);
		}
	}
	stream << desc.str();
}

bool ListReferenceAttributes::isSuperFold(List<MultidimensionalIntervalSeq*> *first, 
		List<MultidimensionalIntervalSeq*> *second) {
	
	if (first == NULL && second != NULL) return false;
	else if (second == NULL) return true;

	int elementsCount = 0;
	int coveredElements = 0;

	for (int i = 0; i < second->NumElements(); i++) {
		MultidimensionalIntervalSeq *otherSeq = second->Nth(i);	
		elementsCount += otherSeq->getNumOfElements();
		for (int j = 0; j < first->NumElements(); j++) {
			MultidimensionalIntervalSeq *mySeq = first->Nth(j);
			List<MultidimensionalIntervalSeq*> *intersect = mySeq->computeIntersection(otherSeq);
			if (intersect == NULL) continue;
			while (intersect->NumElements() > 0) {
				MultidimensionalIntervalSeq *commonPart = intersect->Nth(0);
				coveredElements += commonPart->getNumOfElements();	
				intersect->RemoveAt(0);
				delete commonPart;
			} 
			delete intersect;
		}
	}
	
	return elementsCount == coveredElements;
}

List<MultidimensionalIntervalSeq*> *ListReferenceAttributes::computeSegmentFold(DataPartitionConfig *partConfig, 
                        PartIdContainer *containerTree, std::ofstream *logFile) {
	
	if (containerTree == NULL) return NULL;

	List<PartFolding*> *folds = new List<PartFolding*>;
	containerTree->foldContainer(folds);
	DataItemConfig *dataItemConfig = partConfig->generateStateFulVersion();

	List<MultidimensionalIntervalSeq*> *segmentFold = new List<MultidimensionalIntervalSeq*>;
	for (int i = 0; i < folds->NumElements(); i++) {
		PartFolding *fold = folds->Nth(i);
		List<MultidimensionalIntervalSeq*> *foldDesc 
				= fold->generateIntervalDesc(dataItemConfig, logFile);
		segmentFold->AppendAll(foldDesc);
		delete foldDesc;
	}

	delete dataItemConfig;
	while (folds->NumElements() > 0) {
		PartFolding *fold = folds->Nth(0);
		folds->RemoveAt(0);
		delete fold;
	}
	delete folds;
	return segmentFold;
}

//----------------------------------------------------------- List Reference Key --------------------------------------------------------/

ListReferenceKey::ListReferenceKey(int taskEnvId, const char *varName, const char *allocatorLpsName) {
	this->taskEnvId = taskEnvId;
	this->varName = varName;
	this->allocatorLpsName = allocatorLpsName;
}

char *ListReferenceKey::generateKey() {
	ostringstream stream;
	stream << "env-" << taskEnvId;
	stream << "var-" << varName;
	stream << "space-" << allocatorLpsName;
	return strdup(stream.str().c_str());
}

bool ListReferenceKey::isEqual(ListReferenceKey *other) {
	return (taskEnvId == other->taskEnvId) 
			&& (strcmp(varName, other->varName) == 0)
			&& (strcmp(allocatorLpsName, other->allocatorLpsName) == 0);
}

bool ListReferenceKey::matchesPattern(ListReferenceKey *pattern) {
	if (pattern->taskEnvId != -1 
			&& this->taskEnvId != pattern->taskEnvId) return false;
	if (pattern->varName != NULL 
			&& strcmp(this->varName, pattern->varName) != 0) return false;
	if (pattern->allocatorLpsName != NULL
			&& strcmp(this->allocatorLpsName, pattern->allocatorLpsName) != 0) return false;
	return true;
}

//------------------------------------------------------------ Object Identifier --------------------------------------------------------/

ObjectIdentifier::ObjectIdentifier(int sourceTaskId, int envLinkId) {
	this->sourceTaskId = sourceTaskId;
	this->envLinkId = envLinkId;
}
        
char *ObjectIdentifier::generateKey() {
	ostringstream stream;
	stream << "task-" << sourceTaskId;
	stream << "link-" << envLinkId;
	return strdup(stream.str().c_str());
}

//---------------------------------------------------------- Object Version Manager -----------------------------------------------------/

ObjectVersionManager::ObjectVersionManager(ObjectIdentifier *objectIdentifier, PartsListReference* sourceReference) {
	this->objectIdentifier = objectIdentifier;
	this->freshVersionKeys = new List<ListReferenceKey*>;
	ListReferenceKey *sourceKey = sourceReference->getKey();
	freshVersionKeys->Append(sourceKey);
	dataVersions = new Hashtable<PartsListReference*>;
	dataVersions->Enter(sourceKey->generateKey(), sourceReference);
}

ObjectVersionManager::~ObjectVersionManager() {
	delete objectIdentifier;
	while (freshVersionKeys->NumElements() > 0) {
		ListReferenceKey *key = freshVersionKeys->Nth(0);
		freshVersionKeys->RemoveAt(0);
		delete key;
	}
	delete freshVersionKeys;
	delete dataVersions;
}

void ObjectVersionManager::addNewVersion(PartsListReference *versionReference) {
	ListReferenceKey *sourceKey = versionReference->getKey();
	char *stringKey = sourceKey->generateKey();
	dataVersions->Enter(stringKey, versionReference);
	free(stringKey);
}

void ObjectVersionManager::removeVersion(ListReferenceKey *versionKey) {
	
	char *stringKey = versionKey->generateKey();
	PartsListReference *version = dataVersions->Lookup(stringKey);
	dataVersions->Remove(stringKey, version);
	PartsList *partsList = version->getPartsList();
	delete version;
	free(stringKey);
	
	int referenceCount = partsList->getAttributes()->getReferenceCount();
	if (referenceCount > 1) {
		partsList->getAttributes()->decreaseReferenceCount();
	} else {
		delete partsList;
	}

	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *includedKey = freshVersionKeys->Nth(i);
		if (includedKey->isEqual(versionKey)) {
			freshVersionKeys->RemoveAt(i);
			break;
		}
	}
}

PartsListReference *ObjectVersionManager::getVersion(char *versionKey) {
	return dataVersions->Lookup(versionKey);
}

void ObjectVersionManager::markNonMatchingVersionsStale(ListReferenceKey *matchingKey) {
	
	List<ListReferenceKey*> *updatedList = new List<ListReferenceKey*>;
	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *includedKey = freshVersionKeys->Nth(i);
		if (includedKey->matchesPattern(matchingKey)) {
			updatedList->Append(includedKey);
			continue;
		}
		char *stringKey = includedKey->generateKey();
		PartsListReference *version = dataVersions->Lookup(stringKey);
		PartsList *partsList = version->getPartsList();
		partsList->getAttributes()->flagStale();
		free(stringKey);
	}

	freshVersionKeys->clear();
	freshVersionKeys->AppendAll(updatedList);
	delete updatedList;

	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *freshKey = freshVersionKeys->Nth(i);
		char *stringKey = freshKey->generateKey();
		PartsListReference *version = dataVersions->Lookup(stringKey);
		PartsList *partsList = version->getPartsList();
		partsList->getAttributes()->flagFresh();
		free(stringKey);
	}
}

bool ObjectVersionManager::foundMatchingFreshVersion(ListReferenceKey *matchingKey) {
	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *includedKey = freshVersionKeys->Nth(i);
		if (includedKey->matchesPattern(matchingKey)) {
			return true;
		}
	}
	return false;	
}

void ObjectVersionManager::addFreshVersionKey(ListReferenceKey *freshKey) {
	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *includedKey = freshVersionKeys->Nth(i);
		if (freshKey->isEqual(includedKey)) return;
	}
	freshVersionKeys->Append(freshKey);
}

List<PartsListReference*> *ObjectVersionManager::getFreshVersions() {
	List<PartsListReference*> *freshVersions = new List<PartsListReference*>;
	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *freshKey = freshVersionKeys->Nth(i);
		char *stringKey = freshKey->generateKey();
		freshVersions->Append(dataVersions->Lookup(stringKey));
		free(stringKey);
	}
	return freshVersions;	
}

PartsListReference *ObjectVersionManager::getFirstFreshVersion() {
	ListReferenceKey *firstFreshKey = freshVersionKeys->Nth(0);
	char *stringKey = firstFreshKey->generateKey();
	PartsListReference *version = dataVersions->Lookup(stringKey);
	free(stringKey);
	return version;
}

int ObjectVersionManager::getVersionCount() {
	int versionCount = 0;
	Iterator<PartsListReference*> iterator = dataVersions->GetIterator();
	PartsListReference *version = NULL;
	while ((version = iterator.GetNextValue()) != NULL) {
		versionCount++;
	} 
	return versionCount;
}

//----------------------------------------------------------- Program Environment ------------------------------------------------------/

ProgramEnvironment::ProgramEnvironment() {
	envObjects = new Hashtable<ObjectVersionManager*>;
}

void ProgramEnvironment::addNewDataItem(ObjectIdentifier *identifier, PartsListReference* sourceReference) {
	char *objectKey = identifier->generateKey();
	ObjectVersionManager *versionManager = new ObjectVersionManager(identifier, sourceReference);
	envObjects->Enter(objectKey, versionManager);
	free(objectKey);
}

ObjectVersionManager *ProgramEnvironment::getVersionManager(char *dataItemKey) {
	return envObjects->Lookup(dataItemKey);	
}

void ProgramEnvironment::cleanupPossiblyEmptyVersionManager(char *dataItemKey) {
	ObjectVersionManager *manager = envObjects->Lookup(dataItemKey);
	int versionCount = manager->getVersionCount();
	if (versionCount == 0) {
		envObjects->Remove(dataItemKey, manager);
		delete manager;
	}
}

//---------------------------------------------------------- Environment Link Key ------------------------------------------------------/

EnvironmentLinkKey::EnvironmentLinkKey(const char *varName, int linkId) {
	this->varName = varName;
	this->linkId = linkId;
	this->sourceKey = NULL;
}

void EnvironmentLinkKey::flagAsDataSource(int taskId) {
	sourceKey = generateKey(taskId);
}

char *EnvironmentLinkKey::generateKey(int taskId) {
	ObjectIdentifier identifier = ObjectIdentifier(taskId, linkId);
	return identifier.generateKey();
}

ObjectIdentifier *EnvironmentLinkKey::generateObjectIdentifier(int taskId) {
	return new ObjectIdentifier(taskId, linkId);
}

bool EnvironmentLinkKey::isEqual(EnvironmentLinkKey *other) {
	return (strcmp(this->varName, other->varName) == 0 && this->linkId == other->linkId);
}

//------------------------------------------------------------- LPS Allocation ---------------------------------------------------------/

LpsAllocation::LpsAllocation(const char *lpsId, DataPartitionConfig *partitionConfig) {
	this->lpsId = lpsId;
	this->partitionConfig = partitionConfig;
	this->partContainerTree = NULL;
	this->partsList = NULL;
}

ListReferenceKey *LpsAllocation::generatePartsListReferenceKey(int envId, const char *varName) {
	return new ListReferenceKey(envId, varName, lpsId);
}

PartsListReference *LpsAllocation::generatePartsListReference(int envId, 
		const char *varName, 
		List<Dimension> *rootDimensions) {
	ListReferenceKey *versionKey = generatePartsListReferenceKey(envId, varName);
	ListReferenceAttributes *attributes = new ListReferenceAttributes(partitionConfig, rootDimensions);
	attributes->setPartContainerTree(partContainerTree);
	attributes->computeSegmentFold();
	return new PartsListReference(attributes, versionKey, partsList); 
}

void LpsAllocation::allocatePartsList() {
	List<DataPart*> *dataParts = partsList->getDataParts();
	if (dataParts != NULL) {
		for (int i = 0; i < dataParts->NumElements(); i++) {
			DataPart *dataPart = dataParts->Nth(i);
			dataPart->allocate();
		}
	}
}

//--------------------------------------------------------------- Task Item ------------------------------------------------------------/

TaskItem::TaskItem(EnvironmentLinkKey *key, EnvItemType type, int dimensionality, int elementSize) {	
	this->key = key;
	this->type = type;
	this->rootDimensions = new List<Dimension>;
	for (int i = 0; i < dimensionality; i++) {
		rootDimensions->Append(Dimension());
	}
	allocations = new Hashtable<LpsAllocation*>;
	this->elementSize = elementSize;
	this->environment = NULL;
}

void TaskItem::setEnvironment(TaskEnvironment *environment) {
	this->environment = environment;
}

TaskEnvironment *TaskItem::getEnvironment() { return environment; }

void TaskItem::setRootDimensions(Dimension *dimensions) {
	for (int i = 0; i < rootDimensions->NumElements(); i++) {
		rootDimensions->Nth(i) = dimensions[i];
		rootDimensions->Nth(i).setLength();
	}
}

void TaskItem::setDimension(int dimNo, Dimension dimension) { 
	rootDimensions->RemoveAt(dimNo);
	rootDimensions->InsertAt(dimension, dimNo);
}


void TaskItem::preConfigureLpsAllocation(const char *lpsId, DataPartitionConfig *partitionConfig) {
	LpsAllocation *lpsAllocation = new LpsAllocation(lpsId, partitionConfig);
	allocations->Enter(lpsId, lpsAllocation);
}

bool TaskItem::isEmpty() {
	Iterator<LpsAllocation*>  iterator =  allocations->GetIterator();
	LpsAllocation *allocation = NULL;
	bool noPartsListFound = true;
	while ((allocation = iterator.GetNextValue()) != NULL) {
		if (allocation->getPartsList() != NULL) {
			noPartsListFound = false;
			break;
		}
	}
	return noPartsListFound;
}

const char *TaskItem::getFirstAllocationsLpsId() {
	Iterator<LpsAllocation*>  iterator =  allocations->GetIterator();
	LpsAllocation *allocation = iterator.GetNextValue();
	if (allocation == NULL) return NULL;
	else return allocation->getLpsId();
}

//-------------------------------------------------------- Task Environment -----------------------------------------------------------/

int TaskEnvironment::CURRENT_ENV_ID = 0;

TaskEnvironment::TaskEnvironment() {
	
	this->envId = TaskEnvironment::CURRENT_ENV_ID;
	TaskEnvironment::CURRENT_ENV_ID++;

	this->envItems = new Hashtable<TaskItem*>;
	this->name = NULL;
	this->readersMap = NULL;
	this->writersMap = NULL;
	this->progEnv = NULL;
}

PartReader *TaskEnvironment::getPartReader(const char *itemName, const char *lpsId) {
	ostringstream readerId;
	readerId << itemName << "InSpace" << lpsId << "Reader";
	return readersMap->Lookup(readerId.str().c_str());
}

void TaskEnvironment::setDefaultEnvInitInstrs() {
	Iterator<TaskItem*> iterator = envItems->GetIterator();
	TaskItem *taskItem = NULL;
	while ((taskItem = iterator.GetNextValue()) != NULL) {
		EnvItemType type = taskItem->getType();
		TaskInitEnvInstruction *instr = NULL;
		if (type == IN_OUT) {
			instr = new StaleRefreshInstruction(taskItem);
		} else if (type == OUT) {
			instr = new CreateFreshInstruction(taskItem);
		} else {
			if (taskItem->isEmpty()) {
				instr = new CreateFreshInstruction(taskItem);
			} else {
				instr = new StaleRefreshInstruction(taskItem);
			}
		}
		initInstrs.push_back(instr);
	}
}

void TaskEnvironment::writeItemToFile(const char *itemName, const char *filePath) {
	
	TaskItem *item = envItems->Lookup(itemName);
	const char *allocatorLps = item->getFirstAllocationsLpsId();
	if (allocatorLps == NULL) return;
	
	ostringstream writerId;
	writerId << itemName << "InSpace" << allocatorLps << "Writer";
	PartWriter *writer = writersMap->Lookup(writerId.str().c_str()); 
	if (writer == NULL) return;

	writer->setFileName(filePath);
	writer->processParts();
}	

void TaskEnvironment::addInitEnvInstruction(TaskInitEnvInstruction *instr) { 
	
	TaskItem *itemToUpdate = instr->getItemToUpdate();
	EnvironmentLinkKey *itemKey = itemToUpdate->getEnvLinkKey();
	
	// there must be exactly one initialization instruction per environment item; thus we first check if an instruction is
	// already in the queue for the item
	int oldInstrIndex = - 1;
	for (int i = 0; i < initInstrs.size(); i++) {
		TaskInitEnvInstruction *currInstr = initInstrs[i];
		if (currInstr->getItemToUpdate()->getEnvLinkKey()->isEqual(itemKey)) {
			oldInstrIndex = i;
			break;
		}
	} 
	
	// if there is already an instruction for the item then replace it; otherwise add a new instruction
	if (oldInstrIndex != -1) {
		TaskInitEnvInstruction *oldInstr = initInstrs[oldInstrIndex];
		initInstrs[oldInstrIndex] = instr;
	} else initInstrs.push_back(instr); 
}
        
void TaskEnvironment::addEndEnvInstruction(TaskEndEnvInstruction *instr) { 
	// unlike the case of initialization instructions, there might be multiple task completion instructions for a single item
	endingInstrs.push_back(instr); 
}

TaskInitEnvInstruction *TaskEnvironment::getInstr(const char *itemName, int instrType) {
	for (int i = 0; i < initInstrs.size(); i++) {
		TaskInitEnvInstruction *currInstr = initInstrs[i];
		EnvironmentLinkKey *itemKey = currInstr->getItemToUpdate()->getEnvLinkKey(); 
		if (strcmp(itemKey->getVarName(), itemName) == 0) {
			if (currInstr->getType() == instrType) return currInstr;	
		}
	}	
	return NULL;
}

void TaskEnvironment::setupItemsDimensions() {
	for (int i = 0; i < initInstrs.size(); i++) {
		initInstrs.at(i)->setupDimensions();
	}
}

void TaskEnvironment::preprocessProgramEnvForItems() {
	for (int i = 0; i < initInstrs.size(); i++) {
		initInstrs.at(i)->preprocessProgramEnv();
	}
}

void TaskEnvironment::setupItemsPartsLists() {
	for (int i = 0; i < initInstrs.size(); i++) {
		initInstrs.at(i)->setupPartsList();
	}
}

void TaskEnvironment::postprocessProgramEnvForItems() {
	while (!initInstrs.empty()) {
		TaskInitEnvInstruction *instr = initInstrs.back();
		initInstrs.pop_back();
		instr->postprocessProgramEnv();
		delete instr;
	}
}

void TaskEnvironment::executeTaskCompletionInstructions() {
	while (!endingInstrs.empty()) {
		TaskEndEnvInstruction *instr = endingInstrs.back();
		endingInstrs.pop_back();
		instr->execute();
		delete instr;
	}
}

void TaskEnvironment::resetEnvInstructions() {
	setDefaultEnvInitInstrs();
	setDefaultTaskCompletionInstrs();
}

