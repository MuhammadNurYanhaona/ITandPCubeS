#include "allocation.h"
#include "part_tracking.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../runtime/structure.h"
#include <vector>
#include <cstring>

//---------------------------------------------------------------- Part Metadata ---------------------------------------------------------------/

PartMetadata::PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary, int *padding) {
	Assert(dimensionality > 0 && idList != NULL && boundary != NULL);
	this->dimensionality = dimensionality;
	this->idList = idList;
	this->boundary = boundary;
	this->padding = padding;
}

PartMetadata::~PartMetadata() {
	delete[] boundary;
	delete[] padding;
	while (idList->NumElements() > 0) {
		int *idPart = idList->Nth(0);
		idList->RemoveAt(0);
		delete[] idPart;
	}
	delete idList;
}
        
int PartMetadata::getSize() {
	int size = 1;
	for (int i = 0; i < dimensionality; i++) {
		size *= boundary[i].getLength();
	}
	return size;
}

bool PartMetadata::isMatchingId(List<int*> *candidateId) {
	if (idList->NumElements() != candidateId->NumElements()) return false;
	for (int i = 0; i < idList->NumElements(); i++) {
		int *lpsId1 = idList->Nth(i);
		int *lpsId2 = candidateId->Nth(i); 
		for (int j = 0; j < dimensionality; j++) {
			if (lpsId1[j] != lpsId2[j]) return false;
		}
	}
	return true;
}

void PartMetadata::updateStorageDimension(PartDimension *partDimension) {
	for (int i = 0; i < dimensionality; i++) {
		partDimension[i].storage = boundary[i];
	}
}

//---------------------------------------------------------------- List Metadata ---------------------------------------------------------------/

ListMetadata::ListMetadata(int dimensionality, Dimension *boundary) {
	this->dimensionality = dimensionality;
	this->boundary = boundary;
	this->hasPadding = false;
}

//------------------------------------------------------------------- Data Part ----------------------------------------------------------------/

DataPart::DataPart(PartMetadata *metadata, int epochCount, int elementSize) {
	this->metadata = metadata;
	this->epochCount = epochCount;
	this->dataVersions = new std::vector<void*>;
	dataVersions->reserve(epochCount);
	this->epochHead = 0;
	this->elementSize = elementSize;
}

DataPart::~DataPart() {
	delete metadata;
	for (int i = 0; i < epochCount; i++) {
		void *version = dataVersions->at(i);
		free(version);
	}
	delete dataVersions;
}

void DataPart::allocate(int versionThreshold) {
	
	int size = metadata->getSize();
	int allocationSize = elementSize * size;

	for (int i = versionThreshold; i < epochCount; i++) {
		void *allocation = malloc(sizeof(char) * allocationSize);
		Assert(allocation != NULL);
		char *data = (char *) allocation;
		for (int j = 0; j < allocationSize; j++) {
			data[j] = 0;
		}
		dataVersions->push_back(allocation);
	}
}

void *DataPart::getData() {
	return dataVersions->at(epochHead);
}

void *DataPart::getData(int epoch) {
	int versionIndex = (epochHead + epoch) % epochCount;
	return dataVersions->at(versionIndex);
}

void DataPart::synchronizeAllVersions() {
	
	void *updatedData = dataVersions->at(epochHead);
	int partSize = metadata->getSize() * elementSize;
	
	int epoch = 1;
	while (epoch < epochCount) {
		int versionIndex = (epochHead + epoch) % epochCount;
		void *staleData = dataVersions->at(versionIndex);
		memcpy(staleData, updatedData, partSize);
		epoch++;
	}
}

void DataPart::clone(DataPart *other) {
	
	while (dataVersions->size() > 0) {
		dataVersions->pop_back();
	}	
	
	int currentEpoch = 0;
	while (currentEpoch < other->epochCount) {
		dataVersions->push_back(other->getData(currentEpoch));	
		if (currentEpoch == this->epochCount - 1) break;
		currentEpoch++;
	}

	if (currentEpoch < epochCount - 1) {
		allocate(currentEpoch + 1);
		synchronizeAllVersions();
	}
}

//---------------------------------------------------------------- Data Parts List -------------------------------------------------------------/

DataPartsList::DataPartsList(ListMetadata *metadata, int epochCount) {
	this->partContainer = NULL;
	this->metadata = metadata;
	Assert(epochCount > 0);
	this->epochCount = epochCount;
	this->partList = NULL;
	this->invalid = false;
}

DataPartsList::~DataPartsList() {
	if (!invalid) {
		while (partList->NumElements() > 0) {
			DataPart *part = partList->Nth(0);
			partList->RemoveAt(0);
			delete part;
		}
		delete partList;
	}
}

void DataPartsList::initializePartsList(DataPartitionConfig *partConfig, 
		PartIdContainer *partContainer, 
		int partElementSize) {

	this->partContainer = partContainer;
        int partCount = partContainer->getPartCount();
	if (partCount > 0) {
		partList = new List<DataPart*>(partCount);
		invalid = false;

		PartIterator *iterator = partContainer->getIterator();
		int dimensions = metadata->getDimensions();
		SuperPart *part = NULL;
		int listIndex = 0;

		while ((part = iterator->getCurrentPart()) != NULL) {
			List<int*> *partId = part->getPartId();
			PartLocator *partLocator = new PartLocator(partId, dimensions, listIndex);
			Assert(partLocator != NULL);
			iterator->replaceCurrentPart(partLocator);
			DataPart *dataPart = new DataPart(partConfig->generatePartMetadata(partId),
					epochCount, partElementSize);
			Assert(dataPart != NULL);
			partList->Append(dataPart);
			listIndex++;
			iterator->advance();
		}
	} else {
		invalid = true;
	}

}

void DataPartsList::allocateParts() {
	if (invalid) return;
	for (int i = 0; i < partList->NumElements(); i++) {
		DataPart *dataPart = partList->Nth(i);
		dataPart->allocate();
	}
}


DataPart *DataPartsList::getPart(List<int*> *partId, PartIterator *iterator) {
	SuperPart *part = partContainer->getPart(partId, iterator, metadata->getDimensions());	
	PartLocator *partLocator = reinterpret_cast<PartLocator*>(part);
	int index = partLocator->getPartListIndex();
	return partList->Nth(index);
}

PartIterator *DataPartsList::createIterator() {
	return (partContainer == NULL) ? NULL : partContainer->getIterator();
}
