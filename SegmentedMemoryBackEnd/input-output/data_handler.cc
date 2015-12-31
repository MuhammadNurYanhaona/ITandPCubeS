#include "data_handler.h"
#include "stream.h"
#include "stream.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../utils/list.h"
#include "../utils/interval.h"
#include "../codegen/structure.h"
#include "../communication/part_config.h"
#include "../partition-lib/partition.h"

//--------------------------------------------------------------- Part Info --------------------------------------------------------------/

void PartInfo::clear() {
	while (partDimensions->NumElements() > 0) {

		List<Dimension*> *dimensionInfo = partDimensions->Nth(0);
		while (dimensionInfo->NumElements() > 0) {
			Dimension *dimension = dimensionInfo->Nth(0);
			dimensionInfo->RemoveAt(0);
			delete dimension;
		}
		partDimensions->RemoveAt(0);
		delete dimensionInfo;

		List<int> *partCountInfo = partCounts->Nth(0);
		partCounts->RemoveAt(0);
		delete partCountInfo;
		
		List<int> *partIdListInfo = partIdList->Nth(0);
		partIdList->RemoveAt(0);
		delete partIdListInfo;
	}

	if (contentDescription != NULL) {
		while (contentDescription->NumElements() > 0) {
			MultidimensionalIntervalSeq *seq = contentDescription->Nth(0);
			contentDescription->RemoveAt(0);
			delete seq;
		}
		delete contentDescription;
	}
}

void PartInfo::generateContentDescription(DataPartitionConfig *partConfig) {
	
	DataItemConfig *statefulConfig = partConfig->generateStateFulVersion();
	int dimensionality = statefulConfig->getDimensionality();
	statefulConfig->disablePaddingInAllLevels();
	int levels = statefulConfig->getLevels();
	List<List<IntervalSeq*>*> *dimensionalIntervalDesc = new List<List<IntervalSeq*>*>;

	bool generationFailed = false;	
	for (int d = 0; d < dimensionality; d++) {

		// prepare the dimension configurations and populate a hierarchical ID range to be used for interval
		// description generation
		List<Range> *idRangeList = new List<Range>;
		List<int> *dimIdList = partIdList->Nth(d);
		for (int l = 0; l < levels; l++) {
			int id = dimIdList->Nth(l);
			Range idRange = Range(id);
			// set the parent dimension information that the partition instruction at current level 
			// divides and the parts count
                	statefulConfig->adjustDimensionAndPartsCountAtLevel(l, d);
                	// this is probably not needed for correct interval description generation
                	statefulConfig->setPartIdAtLevel(l, d, id);
			idRangeList->Append(idRange);
		}

		// generate a list of linear interval sequences as the description for current dimension of the part 
		PartitionInstr *instr = statefulConfig->getInstruction(levels - 1, d);
		List<IntervalSeq*> *intervalSeqList = new List<IntervalSeq*>;
		instr->getIntervalDescForRangeHierarchy(idRangeList, intervalSeqList);

		// if any dimensional description is empty then the total part is composed of overlapping regions 
		// from other data parts (this can happens in partitions with multi-level paddings)
		if (intervalSeqList->NumElements() == 0) {
			generationFailed = true;
			break;
		}
		
		dimensionalIntervalDesc->Append(intervalSeqList);
		delete idRangeList;	
	}
	
	if (generationFailed) {
		contentDescription = NULL;
	} else {
		contentDescription = MultidimensionalIntervalSeq::generateIntervalSeqs(dimensionality, 
				dimensionalIntervalDesc);
	}

	// delete the lists holding 1D interval sequences but not the sequences themselves as they remain as parts
        // of the multidimensional interval sequences
        while (dimensionalIntervalDesc->NumElements() > 0) {
                List<IntervalSeq*> *oneDSeqList = dimensionalIntervalDesc->Nth(0);
                dimensionalIntervalDesc->RemoveAt(0);
                delete oneDSeqList;
        }
        delete dimensionalIntervalDesc;
	delete statefulConfig;
}

bool PartInfo::isDataIndexInCorePart(List<int> *dataIndex) { 
	for (int i = 0; i < contentDescription->NumElements(); i++) {
		if (contentDescription->Nth(i)->contains(dataIndex)) return true;
	}
	return false; 
}

//------------------------------------------------------------- Part Handler -------------------------------------------------------------/

PartHandler::PartHandler(DataPartsList *partsList, const char *fileName, DataPartitionConfig *partConfig) {
	this->fileName = fileName;
	this->dataParts = partsList->getPartList();
	this->partConfig = partConfig;
	this->currentPart = NULL;
	this->currentPartInfo = NULL;
	this->currentDataIndex = new List<int>;
	ListMetadata *metadata = partsList->getMetadata();
	this->dataDimensionality = metadata->getDimensions();
	this->dataDimensions = metadata->getBoundary();
	this->needToExcludePadding = false;
}

List<Dimension*> *PartHandler::getDimensionList() {
	List<Dimension*> *dimensionList = new List<Dimension*>;
	for (int i = 0; i < dataDimensionality; i++) dimensionList->Append(&dataDimensions[i]);
	return dimensionList;
}

List<int> *PartHandler::getDataIndex(List<int> *partIndex) {
	
	while (currentDataIndex->NumElements() > 0) currentDataIndex->RemoveAt(0);
        
	int position = currentPart->getMetadata()->getIdList()->NumElements() - 1;
	
	for (int i = 0; i < dataDimensionality; i++) {
		DimPartitionConfig *dimConfig = partConfig->getDimensionConfig(i);
		int dimIndex = partIndex->Nth(i);
		List<int> *partIdList = currentPartInfo->partIdList->Nth(i);
		List<int> *partCounts = currentPartInfo->partCounts->Nth(i);
		List<Dimension*> *partDimensions = currentPartInfo->partDimensions->Nth(i);
		int originalDimIndex = dimConfig->getOriginalIndex(dimIndex, position, 
				partIdList, partCounts, partDimensions);

		currentDataIndex->Append(originalDimIndex);				
	}
	return currentDataIndex;
}

void PartHandler::processParts() {
	begin();
	currentPartInfo = new PartInfo();
	for (int i = 0; i < dataParts->NumElements(); i++) {
		
		DataPart *dataPart = dataParts->Nth(i);
		this->currentPart = dataPart;
		calculateCurrentPartInfo();
		
		if (needToExcludePadding && currentPartInfo->contentDescription == NULL) continue;
	
		PartMetadata *metadata = dataPart->getMetadata();
		Dimension *partDimensions = metadata->getBoundary();
		List<int> *partIndexList = new List<int>;
		processPart(partDimensions, 0, partIndexList);
		delete partIndexList;
	}
	delete currentPartInfo;
	terminate();
}

void PartHandler::calculateCurrentPartInfo() {
	
	List<int*> *partIdList = currentPart->getMetadata()->getIdList();
	currentPartInfo->clear();

	for (int i = 0; i < dataDimensionality; i++) {
		
		List<int> *dimIdList = new List<int>;
		for (int j = 0; j < partIdList->NumElements(); j++) {
			dimIdList->Append(partIdList->Nth(j)[i]);
		}
		List<Dimension*> *dimList = new List<Dimension*>;
		List<int> *dimCountList = new List<int>;
		
		int position = partIdList->NumElements() - 1;
		DimPartitionConfig *dimConfig = partConfig->getDimensionConfig(i);
		dimConfig->getHierarchicalDimensionAndPartCountInfo(dimList, dimCountList, position, dimIdList);
		
		currentPartInfo->partIdList->Append(dimIdList);
		currentPartInfo->partCounts->Append(dimCountList);
		currentPartInfo->partDimensions->Append(dimList);
	}
	
	if (needToExcludePadding) {
		currentPartInfo->generateContentDescription(partConfig);
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
			List<int> *dataIndex = getDataIndex(partialIndex);
			if (!needToExcludePadding || currentPartInfo->isDataIndexInCorePart(dataIndex)) {
				int storeIndex = getStorageIndex(partialIndex, partDimensions);
				processElement(dataIndex, storeIndex, partStore);
			}
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
