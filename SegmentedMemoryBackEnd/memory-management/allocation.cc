#include "allocation.h"
#include "../utils/utility.h"
#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../codegen/structure.h"

//---------------------------------------------------------------- Part Metadata ---------------------------------------------------------------/

PartMetadata::PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary, int *padding) {
	Assert(dimensionality > 0 && idList != NULL && boundary != NULL);
	this->dimensionality = dimensionality;
	this->idList = idList;
	this->boundary = boundary;
	this->padding = padding;
}
        
void PartMetadata::setIntervals(HyperplaneInterval *coreInterval, HyperplaneInterval *paddedInterval) {
	this->coreInterval = coreInterval;
	this->paddedInterval = paddedInterval;
	if (paddedInterval == NULL) this->paddedInterval = coreInterval;
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
		for (int j = 0; j < dimensionality; i++) {
			if (lpsId1[j] != lpsId2[j]) return false;
		}
	}
	return true;
}

//---------------------------------------------------------------- List Metadata ---------------------------------------------------------------/

ListMetadata::ListMetadata(int dimensionality, Dimension *boundary) {
	this->dimensionality = dimensionality;
	this->boundary = boundary;
	this->hasPadding = false;
	this->intervalSpec = NULL;
	this->paddedIntervalSpec = NULL;
}

void ListMetadata::generateIntervalSpec(List<PartMetadata*> *partList) {
	intervalSpec = new IntervalSet;
	intervalSpec->add(partList->Nth(0)->getCoreInterval());
	for (int i = 1; i < partList->NumElements(); i++) {
		IntervalSet *dataSpec = new IntervalSet;
		IntervalSet *currentSpec = intervalSpec;
		dataSpec->add(partList->Nth(i)->getCoreInterval());
		intervalSpec = currentSpec->getUnion(dataSpec);
		delete currentSpec;
		delete dataSpec;
	}
	if (!hasPadding) {
		paddedIntervalSpec = intervalSpec;
	} else {
		paddedIntervalSpec = new IntervalSet;
		paddedIntervalSpec->add(partList->Nth(0)->getPaddedInterval());
		for (int i = 1; i < partList->NumElements(); i++) {
			IntervalSet *dataSpec = new IntervalSet;
			IntervalSet *currentSpec = paddedIntervalSpec;
			dataSpec->add(partList->Nth(i)->getPaddedInterval());
			paddedIntervalSpec = currentSpec->getUnion(dataSpec);
			delete currentSpec;
			delete dataSpec;
		}
	}
}

//---------------------------------------------------------------- Data Parts List -------------------------------------------------------------/

DataPartsList::DataPartsList(ListMetadata *metadata, int epochCount) {
	this->metadata = metadata;
	this->epochCount = epochCount;
	this->partLists = new List<DataPart*>*[epochCount];
	for (int i = 0; i < epochCount; i++) {
		partLists[i] = new List<DataPart*>;
	}
	this->epochHead = 0;
}

DataPart *DataPartsList::getPart(List<int*> *partId) {
	List<DataPart*> *currentList = partLists[epochHead];
	for (int i = 0; i < currentList->NumElements(); i++) {
		DataPart *currentPart = currentList->Nth(i);
		if (currentPart->getMetadata()->isMatchingId(partId)) return currentPart;
	}
	return NULL;
}

DataPart *DataPartsList::getPart(List<int*> *partId, int epoch) {
	int epochVersion = (epochHead - epoch) % epochCount;
	List<DataPart*> *currentList = partLists[epochVersion];
	for (int i = 0; i < currentList->NumElements(); i++) {
		DataPart *currentPart = currentList->Nth(i);
		if (currentPart->getMetadata()->isMatchingId(partId)) return currentPart;
	}
	return NULL;
}
