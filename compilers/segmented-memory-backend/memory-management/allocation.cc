#include "allocation.h"
#include "part_tracking.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../codegen/structure.h"
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

DataPart::DataPart(PartMetadata *metadata, int epochCount) {
	this->metadata = metadata;
	this->epochCount = epochCount;
	this->dataVersions = new std::vector<void*>;
	dataVersions->reserve(epochCount);
	this->epochHead = 0;
	this->elementSize = 0;
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

//---------------------------------------------------------------- Data Parts List -------------------------------------------------------------/

DataPartsList::DataPartsList(ListMetadata *metadata, int epochCount) {
	this->partContainer = NULL;
	this->metadata = metadata;
	Assert(epochCount > 0);
	this->epochCount = epochCount;
	this->partList = NULL;
	this->invalid = false;
}

DataPart *DataPartsList::getPart(List<int*> *partId, PartIterator *iterator) {
	SuperPart *part = partContainer->getPart(partId, iterator, metadata->getDimensions());	
	PartLocator *partLocator = reinterpret_cast<PartLocator*>(part);
	int index = partLocator->getPartListIndex();
	return partList->Nth(index);
}
