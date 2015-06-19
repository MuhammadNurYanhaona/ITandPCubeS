#include "data_handler.h"
#include "stream.h"
#include "stream.h"
#include "../memory-management/allocation.h"
#include "../utils/list.h"
#include "../codegen/structure.h"

//------------------------------------------------------------- Part Handler -------------------------------------------------------------/

PartHandler::PartHandler(DataPartsList *partsList, const char *fileName) {
	this->fileName = fileName;
	this->dataParts = partsList->getCurrentList();
	this->currentPart = NULL;
	ListMetadata *metadata = partsList->getMetadata();
	this->dataDimensionality = metadata->getDimensions();
	this->dataDimensions = metadata->getBoundary();
}

List<Dimension*> *PartHandler::getDimensionList() {
	List<Dimension*> *dimensionList = new List<Dimension*>;
	for (int i = 0; i < dataDimensionality; i++) dimensionList->Append(&dataDimensions[i]);
	return dimensionList;
}

void PartHandler::processParts() {
	begin();
	for (int i = 0; i < dataParts->NumElements(); i++) {
		DataPart *dataPart = dataParts->Nth(i);
		this->currentPart = dataPart;
		PartMetadata *metadata = dataPart->getMetadata();
		Dimension *partDimensions = metadata->getBoundary();
		List<int> *partIndexList = new List<int>;
		processPart(partDimensions, 0, partIndexList);
	}
	terminate();
}

void PartHandler::processPart(Dimension *partDimensions, int currentDimNo, List<int> *partialIndex) {
	void *partStore = getCurrentPartData();
	Dimension dimension = partDimensions[currentDimNo];
	for (int index = dimension.range.min; index <= dimension.range.max; index++) {
		partialIndex->Append(index);
		if (currentDimNo < dataDimensionality - 1) {
			processPart(partDimensions, currentDimNo + 1, partialIndex);	
		} else {
			int storeIndex = getStorageIndex(partialIndex, partDimensions);
			List<int> *dataIndex = getDataIndex(partialIndex);
			processElement(dataIndex, storeIndex, partStore);
		}
		partialIndex->RemoveAt(currentDimNo);
	}
}

int PartHandler::getStorageIndex(List<int> *partIndex, Dimension *partDimensions) {
	int storeIndex = 0;
	int multiplier = 1;
	for (int i = partIndex->NumElements() - 1; i >= 0; i--) {
		int firstIndex = partDimensions[i].range.min;
		int dimensionIndex = partIndex->Nth(i);
		storeIndex += (dimensionIndex - firstIndex) * multiplier;
		multiplier *= partDimensions[i].length;
	}
	return storeIndex;
}
