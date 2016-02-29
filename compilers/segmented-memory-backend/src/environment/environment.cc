#include "environment.h"
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
	delete partitionConfig;
	delete rootDimensions;
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

void ListReferenceAttributes::computeSegmentFold() {
	
	if (partContainerTree == NULL) {
		segmentFold = NULL;
		return;
	}

	List<PartFolding*> *folds = new List<PartFolding*>;
	partContainerTree->foldContainer(folds);
	DataItemConfig *dataItemConfig = partitionConfig->generateStateFulVersion();

	segmentFold = new List<MultidimensionalIntervalSeq*>;
	for (int i = 0; i < folds->NumElements(); i++) {
		PartFolding *fold = folds->Nth(i);
		List<MultidimensionalIntervalSeq*> *foldDesc = fold->generateIntervalDesc(dataItemConfig);
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
}

bool ListReferenceAttributes::isSuperFold(List<MultidimensionalIntervalSeq*> *otherFold) {
	
	if (segmentFold == NULL && otherFold != NULL) return false;
	else if (otherFold == NULL) return true;

	int elementsCount = 0;
	int coveredElements = 0;

	for (int i = 0; i < otherFold->NumElements(); i++) {
		MultidimensionalIntervalSeq *otherSeq = otherFold->Nth(i);	
		elementsCount += otherSeq->getNumOfElements();
		for (int j = 0; j < segmentFold->NumElements(); j++) {
			MultidimensionalIntervalSeq *mySeq = segmentFold->Nth(j);
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

//----------------------------------------------------------- List Reference Key --------------------------------------------------------/

ListReferenceKey::ListReferenceKey(int taskEnvId, const char *varName, const char *allocatorLpsName) {
	this->taskEnvId = taskEnvId;
	this->varName = varName;
	this->allocatorLpsName = allocatorLpsName;
}

const char *ListReferenceKey::generateKey() {
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
        
const char *ObjectIdentifier::generateKey() {
	ostringstream stream;
	stream << "task-" << sourceTaskId;
	stream << "link-" << envLinkId;
	return strdup(stream.str().c_str());
}

//---------------------------------------------------------- Object Version Manager -----------------------------------------------------/

ObjectVersionManager::ObjectVersionManager(ObjectIdentifier *objectIdentifier, PartsListReference* sourceReference) {
	this->objectIdentifer = objectIdentifier;
	this->freshVersionKeys = new List<ListReferenceKey*>;
	ListReferenceKey *sourceKey = sourceReference->getKey();
	freshVersionKeys->Append(sourceKey);
	dataVersions = new Hashtable<PartsListReference*>;
	dataVersions->Enter(sourceKey->generateKey(), sourceReference);
}

ObjectVersionManager::~ObjectVersionManager() {
	delete objectIdentifer;
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
	const char *stringKey = sourceKey->generateKey();
	dataVersions->Enter(stringKey, versionReference);
}

void ObjectVersionManager::removeVersion(ListReferenceKey *versionKey) {
	
	const char *stringKey = versionKey->generateKey();
	PartsListReference *version = dataVersions->Lookup(stringKey);
	PartsList *partsList = version->getPartsList();
	delete version;
	
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

PartsListReference *ObjectVersionManager::getVersion(const char *versionKey) {
	return dataVersions->Lookup(versionKey);
}

void ObjectVersionManager::markNonMatchingVersionsStale(ListReferenceKey *matchingKey) {
	List<ListReferenceKey*> updatedList;
	for (int i = 0; i < freshVersionKeys->NumElements(); i++) {
		ListReferenceKey *includedKey = freshVersionKeys->Nth(i);
		if (includedKey->matchesPattern(matchingKey)) {
			updatedList.Append(includedKey);
			continue;
		}
		const char *stringKey = includedKey->generateKey();
		PartsListReference *version = dataVersions->Lookup(stringKey);
		PartsList *partsList = version->getPartsList();
		partsList->getAttributes()->flagStale();
		delete stringKey;
	}
	freshVersionKeys->clear();
	freshVersionKeys->AppendAll(&updatedList);
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
		const char *stringKey = freshKey->generateKey();
		freshVersions->Append(dataVersions->Lookup(stringKey));
		delete stringKey;
	}
	return freshVersions;	
}

//----------------------------------------------------------- Program Environment ------------------------------------------------------/

ProgramEnvironment::ProgramEnvironment() {
	envObjects = new Hashtable<ObjectVersionManager*>;
}

void ProgramEnvironment::addNewDataItem(ObjectIdentifier *identifier, PartsListReference* sourceReference) {
	const char *objectKey = identifier->generateKey();
	ObjectVersionManager *versionManager = new ObjectVersionManager(identifier, sourceReference);
	envObjects->Enter(objectKey, versionManager);
}

ObjectVersionManager *ProgramEnvironment::getVersionManager(const char *dataItemKey) {
	return envObjects->Lookup(dataItemKey);	
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

const char *EnvironmentLinkKey::generateKey(int taskId) {
	ostringstream stream;
	stream << "task-" << taskId;
	stream << "link-" << linkId;
	return strdup(stream.str().c_str());
}

ObjectIdentifier *EnvironmentLinkKey::generateObjectIdentifier(int taskId) {
	return new ObjectIdentifier(taskId, linkId);
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

//--------------------------------------------------------------- Task Item ------------------------------------------------------------/

TaskItem::TaskItem(EnvironmentLinkKey *key, int dimensionality) {
	this->key = key;
	this->rootDimensions = new List<Dimension>;
	for (int i = 0; i < dimensionality; i++) {
		rootDimensions->Append(Dimension());
	}
	allocations = new Hashtable<LpsAllocation*>;
}

void TaskItem::setRootDimensions(Dimension *dimensions) {
	for (int i = 0; i < rootDimensions->NumElements(); i++) {
		rootDimensions->Nth(i) = dimensions[i];
		rootDimensions->Nth(i).setLength();
	}
}

void TaskItem::preConfigureLpsAllocation(const char *lpsId, DataPartitionConfig *partitionConfig) {
	LpsAllocation *lpsAllocation = new LpsAllocation(lpsId, partitionConfig);
	allocations->Enter(lpsId, lpsAllocation);
}
