#include "part_management.h"
#include "part_tracking.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../runtime/structure.h"
#include "part_generation.h"
#include "allocation.h"

#include <cstdlib>
#include <sstream>

//--------------------------------------------------------------- Data Items ---------------------------------------------------------------/

DataItems::DataItems(const char *name, int dimensionality) {
	this->name = name;
	this->dimensionality = dimensionality;
	this->ready = false;
	this->dimConfigList = new List<DimPartitionConfig*>;
	Assert(this->dimConfigList != NULL);
	this->partitionConfig = NULL;
	this->partsList = NULL;
}

void DataItems::addDimPartitionConfig(int dimensionId, DimPartitionConfig *dimConfig) {
	Assert(dimensionId < dimensionality);
	if (dimConfigList->NumElements() > dimensionId) {
		dimConfigList->InsertAt(dimConfig, dimensionId);
	} else dimConfigList->Append(dimConfig);
}

void DataItems::generatePartitionConfig() {
	Assert(dimConfigList->NumElements() == dimensionality);
	partitionConfig = new DataPartitionConfig(dimensionality, dimConfigList);
	Assert(partitionConfig != NULL);
	ready = true;
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
	Assert(lpsContentMap != NULL);
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


