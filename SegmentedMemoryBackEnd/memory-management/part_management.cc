#include "part_management.h"
#include "part_tracking.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "part_generation.h"
#include "allocation.h"

//--------------------------------------------------------------- Data Items ---------------------------------------------------------------/

DataItems::DataItems(const char *name, int dimensionality, int epochCount) {
	this->name = name;
	this->dimensionality = dimensionality;
	this->epochCount = epochCount;
	this->ready = false;
	this->dimConfigList = new List<DimPartitionConfig*>;
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
	return partsList->createIterator();
}

DataPart *DataItems::getDataPart(List<int*> *partIdList, PartIterator *iterator) {
	Assert(partsList != NULL);
	return partsList->getPart(partIdList, iterator);
}

DataPart *DataItems::getDataPart(List<int*> *partIdList, int epoch, PartIterator *iterator) {
	Assert(partsList != NULL);
	return partsList->getPart(partIdList, epoch, iterator);
}
        
List<DataPart*> *DataItems::getAllDataParts() {
	Assert(partsList != NULL);
	return partsList->getCurrentList();
}

//---------------------------------------------------------- Scalar Data Items -------------------------------------------------------------/

ScalarDataItems::ScalarDataItems(const char *name, int epochCount) : DataItems(name, 0, epochCount) {
	variableList = NULL;
	epochHead = 0;
}

void *ScalarDataItems::getVariable() { 
	Assert(ready);
	return variableList[epochHead]; 
}
        
void *ScalarDataItems::getVariable(int version) {
	Assert(ready);
	int versionEpoch = (epochHead - version) % epochCount;
	return variableList[versionEpoch];
}

//------------------------------------------------------------ LPS Content ----------------------------------------------------------------/

LpsContent::LpsContent(int id) {
	this->id = id;
	this->dataItemsMap = new Hashtable<DataItems*>;
}

void LpsContent::advanceItemEpoch(const char *varName) {
	DataItems *dataItems = dataItemsMap->Lookup(varName);
	if (dataItems != NULL) {
		dataItems->advanceEpoch();
	}
}

//------------------------------------------------------------- Task Data ----------------------------------------------------------------/

TaskData::TaskData() { lpsContentMap = new Hashtable<LpsContent*>; }

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

void TaskData::advanceItemEpoch(const char *lpsId, const char *varName) {
	LpsContent *lpsContent = lpsContentMap->Lookup(lpsId);
	lpsContent->advanceItemEpoch(varName);
}


