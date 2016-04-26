#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include "data_transfer.h"
#include "part_config.h"
#include "../runtime/structure.h"
#include "../memory-management/allocation.h"
#include "../utils/utility.h"

using namespace std;

//---------------------------------------------------- Transfer Specification -----------------------------------------------------/

TransferSpec::TransferSpec(TransferDirection direction, int elementSize) {
	this->direction = direction;
	this->elementSize = elementSize;
	this->bufferEntry = NULL;
	this->dataIndex = NULL;
}

void TransferSpec::setBufferEntry(char *bufferEntry, vector<int> *dataIndex) {
	this->bufferEntry = bufferEntry;
	this->dataIndex = dataIndex;
}

void TransferSpec::performTransfer(char *dataPartLocation) {
	if (direction == COMM_BUFFER_TO_DATA_PART) {
		memcpy(dataPartLocation, bufferEntry, elementSize);
	} else {
		memcpy(bufferEntry, dataPartLocation, elementSize);
	}
}

//--------------------------------------------------- Data Part Specification -----------------------------------------------------/

DataPartSpec::DataPartSpec(List<DataPart*> *partList, DataItemConfig *dataConfig) {
	this->partList = partList;
	this->dataConfig = dataConfig;
	dimensionality = dataConfig->getDimensionality();
	dataDimensions = new Dimension[dimensionality];
	for (int i = 0; i < dimensionality; i++) {
		dataDimensions[i] = dataConfig->getDimension(i);
	}
}

void DataPartSpec::initPartTraversalReference(vector<int> *dataIndex, vector<XformedIndexInfo*> *transformVector) {
	for (int i = 0; i < dimensionality; i++) {
		XformedIndexInfo *dimIndex = transformVector->at(i);
		dimIndex->index = dataIndex->at(i);
		dimIndex->partNo = 0;
		dimIndex->partDimension = dataDimensions[i];
	}
}

char *DataPartSpec::getUpdateLocation(PartLocator *partLocator, vector<int> *partIndex, int dataItemSize) {

	int partNo = partLocator->getPartListIndex();
	DataPart *dataPart = partList->Nth(partNo);
	PartMetadata *metadata = dataPart->getMetadata();
	Dimension *partDimensions = metadata->getBoundary();

	int dataPointNo = 0;
	int multiplier = 1;
	for (int i = partIndex->size() - 1; i >= 0; i--) {
		
		int firstIndex = partDimensions[i].range.min;
		int lastIndex = partDimensions[i].range.max;
		int dimensionIndex = partIndex->at(i);
		
		Assert(firstIndex <= dimensionIndex && dimensionIndex <= lastIndex);
		
		dataPointNo += (dimensionIndex - firstIndex) * multiplier;
		multiplier *= partDimensions[i].length;
	}

	void *data = dataPart->getData();
	char *charData = reinterpret_cast<char*>(data);
	
	Assert(dataPointNo < metadata->getSize());

	char *updateLocation = charData + dataItemSize * dataPointNo;
	return updateLocation;
}
