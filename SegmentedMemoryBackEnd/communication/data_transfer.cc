#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include "data_transfer.h"
#include "part_config.h"
#include "../codegen/structure.h"
#include "../memory-management/allocation.h"

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

DataPartSpec::DataPartSpec(DataPartsList *dataParts, DataItemConfig *dataConfig) {
	this->dataParts = dataParts;
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
	List<DataPart*> *partList = dataParts->getPartList();
	DataPart *dataPart = partList->Nth(partNo);
	Dimension *partDimensions = dataPart->getMetadata()->getBoundary();

	int dataPointNo = 0;
	int multiplier = 1;
	for (int i = partIndex->size() - 1; i >= 0; i--) {
		int firstIndex = partDimensions[i].range.min;
		int dimensionIndex = partIndex->at(i);
		dataPointNo += (dimensionIndex - firstIndex) * multiplier;
		multiplier *= partDimensions[i].length;
	}

	void *data = dataPart->getData();
	char *charData = reinterpret_cast<char*>(data);

	char *updateLocation = charData + dataItemSize * dataPointNo;
	return updateLocation;
}
