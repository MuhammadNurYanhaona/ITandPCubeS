#include "allocation.h"
#include "part_tracking.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../codegen/structure.h"

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

//---------------------------------------------------------------- Data Parts List -------------------------------------------------------------/

DataPartsList::DataPartsList(ListMetadata *metadata, int epochCount) {
	this->partContainer = NULL;
	this->metadata = metadata;
	this->epochCount = epochCount;
	Assert(epochCount > 0);
	this->partLists = new List<DataPart*>*[epochCount];
	Assert(this->partLists != NULL);
	for (int i = 0; i < epochCount; i++) {
		partLists[i] = NULL;
	}
	this->epochHead = 0;
}

void DataPartsList::allocateLists(int capacity) {
	for (int i = 0; i < epochCount; i++) {
		partLists[i] = new List<DataPart*>(capacity);
		Assert(partLists[i] != NULL);
	}
}

DataPart *DataPartsList::getPart(List<int*> *partId, PartIterator *iterator) {
	SuperPart *part = partContainer->getPart(partId, iterator, metadata->getDimensions());	
	List<DataPart*> *currentList = partLists[epochHead];
	PartLocator *partLocator = reinterpret_cast<PartLocator*>(part);
	int index = partLocator->getPartListIndex();
	return currentList->Nth(index);
}

DataPart *DataPartsList::getPart(List<int*> *partId, int epoch, PartIterator *iterator) {
	int epochVersion = (epochHead - epoch) % epochCount;
	List<DataPart*> *currentList = partLists[epochVersion];
	SuperPart *part = partContainer->getPart(partId, iterator, metadata->getDimensions());	
	PartLocator *partLocator = reinterpret_cast<PartLocator*>(part);
	int index = partLocator->getPartListIndex();
	return currentList->Nth(index);
}

