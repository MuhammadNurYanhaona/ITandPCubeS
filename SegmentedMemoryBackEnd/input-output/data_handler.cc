#include "data_handler.h"
#include "stream.h"
#include "stream.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../utils/list.h"
#include "../codegen/structure.h"

//------------------------------------------------------------- Part Handler -------------------------------------------------------------/

PartHandler::PartHandler(DataPartsList *partsList, const char *fileName, DataPartitionConfig *partConfig) {
	this->fileName = fileName;
	this->dataParts = partsList->getCurrentList();
	this->partConfig = partConfig;
	this->currentPart = NULL;
	this->currentPartInfo = NULL;
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
		calculateCurrentPartInfo();
	
		PartMetadata *metadata = dataPart->getMetadata();
		Dimension *partDimensions = metadata->getBoundary();
		List<int> *partIndexList = new List<int>;
		processPart(partDimensions, 0, partIndexList);
	}
	terminate();
}

void PartHandler::calculateCurrentPartInfo() {
	
	List<int*> *partIdList = currentPart->getMetadata()->getIdList();
	currentPartInfo = new PartInfo();
	for (int i = 0; i < dataDimensionality; i++) {
		
		List<int> *dimIdList = new List<int>;
		for (int j = 0; j < partIdList->NumElements(); j++) {
			dimIdList->Append(partIdList->Nth(j)[i]);
		}
		List<Dimension> *dimList = new List<Dimension>;
		List<int> *dimCountList = new List<int>;
		
		int position = partIdList->NumElements() - 1;
		DimPartitionConfig *dimConfig = partConfig->getDimensionConfig(i);
		dimConfig->getHierarchicalDimensionAndPartCountInfo(dimList, dimCountList, position, dimIdList);
		
		currentPartInfo->partIdList->Append(dimIdList);
		currentPartInfo->partCounts->Append(dimCountList);
		currentPartInfo->partDimensions->Append(dimList);
	}
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
