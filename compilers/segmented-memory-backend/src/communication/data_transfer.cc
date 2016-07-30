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

//----------------------------------------------------- Data Part Index List ------------------------------------------------------/

void DataPartIndexList::clone(DataPartIndexList *source) {
	this->partIndexList->clear();
	this->partIndexList->AppendAll(source->partIndexList);
}

int DataPartIndexList::read(char *destBuffer, int elementSize) {
	
	// Notice that data is read from only the first matched location in the operating memory data part storage. 
	// This is because, all locations matching a single entry in the communication buffer should have identical 
	// content.
	char *readLocation = partIndexList->Nth(0).getLocation();
        memcpy(destBuffer, readLocation, elementSize);
	return 1;
}
        
int DataPartIndexList::write(char *sourceBuffer, int elementSize) {

	// Unlike in the case for read, writing should access every single matched location in the data parts as we 
	// need to ensure that all locations matching a single entry in the communication buffer are synchronized 
	// with the same update.       
	for (int j = 0; j < partIndexList->NumElements(); j++) {
		char *writeLocation = partIndexList->Nth(j).getLocation();
		memcpy(writeLocation, sourceBuffer, elementSize);
	}

	// The return value is still once as only one element has been read from the source buffer
	return 1;
}

//-------------------------------------------------- Data Part Swift Index List ---------------------------------------------------/

DataPartSwiftIndexList::DataPartSwiftIndexList(DataPart *dataPart) : DataPartIndexList() {
	this->dataPart = dataPart;
	partIndexes = new List<int>;
} 

DataPartSwiftIndexList::~DataPartSwiftIndexList() {
	delete partIndexes;
	delete[] indexArray;
}

void DataPartSwiftIndexList::setupIndexArray() {
	int indexCount = partIndexes->NumElements();
	indexArray = new int[indexCount];
	for (int i = 0; i < indexCount; i++) {
		indexArray[i] = partIndexes->Nth(i);
	}
	sequenceLength = indexCount;
}

int DataPartSwiftIndexList::read(char *destBuffer, int elementSize) {
	void *data = dataPart->getData();
        char *charData = reinterpret_cast<char*>(data);
	char *currBufferIndex = destBuffer;
	for (int i = 0; i < sequenceLength; i++) {
		char *readLocation = charData + indexArray[i];
		memcpy(currBufferIndex, readLocation, elementSize);
		currBufferIndex += elementSize;
	}
	return sequenceLength;
}
        
int DataPartSwiftIndexList::write(char *sourceBuffer, int elementSize) {
	void *data = dataPart->getData();
        char *charData = reinterpret_cast<char*>(data);
	char *currBufferIndex = sourceBuffer;
	for (int i = 0; i < sequenceLength; i++) {
		char *writeLocation = charData + indexArray[i];
		memcpy(writeLocation, currBufferIndex, elementSize);
		currBufferIndex += elementSize;
	}
	return sequenceLength;
}

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

void TransferSpec::performTransfer(DataPartIndex dataPartIndex) {
	char *dataPartLocation = dataPartIndex.getLocation();
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

	DataPartIndex dataPartIndex = getDataPartUpdateIndex(partLocator, partIndex, dataItemSize);
	return dataPartIndex.getLocation();
}

DataPartIndex DataPartSpec::getDataPartUpdateIndex(PartLocator *partLocator, 
		vector<int> *partIndex, int dataItemSize) {

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

        Assert(dataPointNo < metadata->getSize());
	DataPartIndex dataPartIndex = DataPartIndex(dataPart, dataItemSize * dataPointNo);
	return dataPartIndex;
}
