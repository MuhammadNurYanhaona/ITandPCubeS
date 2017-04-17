#include "part_management.h"
#include "part_tracking.h"
#include "part_generation.h"
#include "allocation.h"

#include "../../../../common-libs/utils/utility.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/domain-obj/structure.h"

#include <cstdlib>
#include <sstream>

//--------------------------------------------------------------- Data Items ---------------------------------------------------------------/

DataItems::DataItems(const char *name, int dimensionality, bool cleanable) {
	this->name = name;
	this->dimensionality = dimensionality;
	this->ready = false;
	this->partitionConfig = NULL;
	this->partsList = NULL;
	this->cleanable = cleanable;
}

DataItems::~DataItems() {
	if (cleanable) {
		delete partsList;
		delete partitionConfig;
	}
}

void DataItems::setPartitionConfig(DataPartitionConfig *partitionConfig) {
	ready = true;
	this->partitionConfig = partitionConfig;
}

DataPartitionConfig *DataItems::getPartitionConfig() { 
	Assert(ready);
	return partitionConfig; 
}

PartIterator *DataItems::createIterator() {
	Assert(partsList != NULL);
	if (partsList->isInvalid()) return NULL;
	return partsList->createIterator();
}

DataPart *DataItems::getDataPart(List<int*> *partIdList, PartIterator *iterator) {
	Assert(partsList != NULL);
	return partsList->getPart(partIdList, iterator);
}

List<DataPart*> *DataItems::getAllDataParts() {
	Assert(partsList != NULL);
	return partsList->getPartList();
}

//------------------------------------------------------------ LPS Content ----------------------------------------------------------------/

LpsContent::LpsContent(int id) {
	this->id = id;
	this->dataItemsMap = new Hashtable<DataItems*>;
	Assert(this->dataItemsMap != NULL);
}

LpsContent::~LpsContent() {
	Iterator<DataItems*> iterator = dataItemsMap->GetIterator();
	DataItems *items = NULL;
	List<DataItems*> *itemsList = new List<DataItems*>;
	while ((items = iterator.GetNextValue()) != NULL) {
		itemsList->Append(items);
	}
	while (itemsList->NumElements() > 0) {
		DataItems *nextItem = itemsList->Nth(0);
		itemsList->RemoveAt(0);
		delete nextItem;	
	}
	delete itemsList;
	delete dataItemsMap;
}

void LpsContent::addPartIterators(Hashtable<PartIterator*> *partIteratorMap) {
	Iterator<DataItems*> iterator = dataItemsMap->GetIterator();
	DataItems *items = NULL;
	while ((items = iterator.GetNextValue()) != NULL) {
		PartIterator *iterator = items->createIterator();
		if (iterator == NULL) continue;

		int dimensions = items->getDimensions();
		int partIdLevels = items->getPartitionConfig()->getPartIdLevels();
		iterator->initiatePartIdTemplate(dimensions, partIdLevels);
		std::ostringstream key;
		key << "Space_" << id << "_Var_" << items->getName();
		partIteratorMap->Enter(strdup(key.str().c_str()), iterator);
	}
}

bool LpsContent::hasValidDataItems() {
	Iterator<DataItems*> iterator = dataItemsMap->GetIterator();
	DataItems *items = NULL;
	while ((items = iterator.GetNextValue()) != NULL) {
		if (!items->isEmpty()) return true;
	}
	return false;
}

//------------------------------------------------------------- Task Data ----------------------------------------------------------------/

TaskData::TaskData() { 
	lpsContentMap = new Hashtable<LpsContent*>; 
	reductionResultMap = new Hashtable<ReductionResultAccessContainer*>;
	Assert(lpsContentMap != NULL && reductionResultMap != NULL);
}

TaskData::~TaskData() {
	
	Iterator<LpsContent*> lpsIterator = lpsContentMap->GetIterator();
	LpsContent *lpsContent = NULL;
	List<LpsContent*> *contentList = new List<LpsContent*>;
	while ((lpsContent = lpsIterator.GetNextValue()) != NULL) {
		contentList->Append(lpsContent);
	}
	while (contentList->NumElements() > 0) {
		LpsContent *nextContent = contentList->Nth(0);
		contentList->RemoveAt(0);
		delete nextContent;
	}
	delete contentList;
	delete lpsContentMap;
	lpsContentMap = NULL;

	Iterator<ReductionResultAccessContainer*> reductionIterator 
			= reductionResultMap->GetIterator();
	List<ReductionResultAccessContainer*> *resultList 
			= new List<ReductionResultAccessContainer*>;
	ReductionResultAccessContainer *redResContainer;
	while ((redResContainer = reductionIterator.GetNextValue()) != NULL) {
		resultList->Append(redResContainer);
	}
	while (resultList->NumElements() > 0) {
		ReductionResultAccessContainer *container = resultList->Nth(0);
		resultList->RemoveAt(0);
		delete container;
	}
	delete resultList;
	delete reductionResultMap;
	reductionResultMap = NULL;		
}

void TaskData::addLpsContent(const char *lpsId, LpsContent *content) { 
	if (content != NULL) {
		lpsContentMap->Enter(lpsId, content); 
	}
}

DataItems *TaskData::getDataItemsOfLps(const char *lpsId, const char *varName) {
	LpsContent *lpsContent = lpsContentMap->Lookup(lpsId);
	if (lpsContent == NULL) return NULL;
	else return lpsContent->getDataItems(varName);
}

void TaskData::addReductionResultContainer(const char *varName, ReductionResultAccessContainer *container) {
	reductionResultMap->Enter(varName, container);
}

reduction::Result *TaskData::getResultVar(const char *varName, List<int*> *lpuId) {
	ReductionResultAccessContainer *container = reductionResultMap->Lookup(varName);
	return container->getResultForLpu(lpuId);
}

Hashtable<PartIterator*> *TaskData::generatePartIteratorMap() {
	Hashtable<PartIterator*> *map = new Hashtable<PartIterator*>;
	Assert(map != NULL);
	Iterator<LpsContent*> iterator = lpsContentMap->GetIterator();
	LpsContent *lpsContent = NULL;
	while ((lpsContent = iterator.GetNextValue()) != NULL) {
		lpsContent->addPartIterators(map);
	}
	return map;
}

bool TaskData::hasDataForLps(const char *lpsId) {
	LpsContent *lpsContent = lpsContentMap->Lookup(lpsId);
	if (lpsContent == NULL) return false;
	return lpsContent->hasValidDataItems();
}


