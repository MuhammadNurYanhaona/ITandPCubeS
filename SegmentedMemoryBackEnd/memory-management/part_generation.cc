#include "part_generation.h"
#include "allocation.h"
#include "part_tracking.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"
#include "../codegen/structure.h"
#include "../partition-lib/partition.h"
#include "../communication/part_config.h"

#include <stack>
#include <vector>
#include <algorithm>
#include <iostream>

//---------------------------------------------------------- Dim Partition Config ---------------------------------------------------------/

DimPartitionConfig::DimPartitionConfig(Dimension dimension, int *partitionArgs,
		int paddings[2], int ppuCount, int lpsAlignment) {
	this->dataDimension = dimension;
	this->partitionArgs = partitionArgs;
	this->paddings[0] = paddings[0];
	this->paddings[1] = paddings[1];
	this->ppuCount = ppuCount;
	this->lpsAlignment = lpsAlignment;
	this->parentConfig = NULL;
}

DimPartitionConfig::DimPartitionConfig(Dimension dimension, int *partitionArgs, int ppuCount, int lpsAlignment) {
	this->dataDimension = dimension;
	this->partitionArgs = partitionArgs;
	this->paddings[0] = 0;
	this->paddings[1] = 0;
	this->ppuCount = ppuCount;
	this->lpsAlignment = lpsAlignment;
	this->parentConfig = NULL;
}

Dimension DimPartitionConfig::getDimensionFromParent(List<int> *partIdList, int position) {
	if (parentConfig == NULL) return dataDimension;
	Dimension parentDataDimension = parentConfig->getDimensionFromParent(partIdList, position - 1);
	int parentPartId = partIdList->Nth(position - 1);
	return parentConfig->getPartDimension(parentPartId, parentDataDimension);
}

DimensionMetadata *DimPartitionConfig::generateDimMetadata(List<int> *partIdList) {

	int position = partIdList->NumElements() - 1;
	int partId = partIdList->Nth(position);
	Dimension parentDimension = getDimensionFromParent(partIdList, position);

	DimensionMetadata *metadata = new DimensionMetadata();
	Assert(metadata != NULL);
	metadata->partDimension = getPartDimension(partId, parentDimension);
	
	metadata->paddings[0] = 0;
	metadata->paddings[1] = 0;
	if (paddings[0] > 0) metadata->paddings[0] = getEffectiveFrontPadding(partId, parentDimension);
	if (paddings[1] > 0) metadata->paddings[1] = getEffectiveRearPadding(partId, parentDimension);

	return metadata;
}

Dimension DimPartitionConfig::getPartDimension(List<int> *partIdList) {
	int position = partIdList->NumElements() - 1;
	int partId = partIdList->Nth(position);
	Dimension parentDimension = getDimensionFromParent(partIdList, position);
	return getPartDimension(partId, parentDimension);	
}

void DimPartitionConfig::getHierarchicalDimensionAndPartCountInfo(List<Dimension*> *dimensionList,
		List<int> *partCountsList, 
		int position, List<int> *partIdList) {

	int partId = partIdList->Nth(position);
	Dimension *dimension = new Dimension();
	Assert(dimension != NULL);
	if (position == 0) {
		*dimension = getPartDimension(partId, dataDimension);
		dimensionList->Append(dimension);
		partCountsList->Append(getPartsCount(dataDimension));
	} else {
		parentConfig->getHierarchicalDimensionAndPartCountInfo(dimensionList, 
				partCountsList, position - 1, partIdList);
		Dimension *parentDimension = dimensionList->Nth(dimensionList->NumElements() - 1);
		*dimension = getPartDimension(partId, *parentDimension);
		dimensionList->Append(dimension);
		partCountsList->Append(getPartsCount(*parentDimension));
	}
}

int DimPartitionConfig::getOriginalIndex(int partIndex, int position, 
		List<int> *partIdList,
		List<int> *partCountList, 
		List<Dimension*> *partDimensionList) {

	if (position > 0) {
		return parentConfig->getOriginalIndex(partIndex, position - 1, partIdList, 
				partCountList, partDimensionList);
	} else return partIndex;
}

//------------------------------------------------------------ Block Size Config ----------------------------------------------------------/

int BlockSizeConfig::getPartsCount(Dimension parentDimension) {
	int size = partitionArgs[0];
	int dimLength = parentDimension.length;
	return (dimLength + size - 1) / size;
}

Dimension BlockSizeConfig::getPartDimension(int partId, Dimension parentDimension) {
	
	int size = partitionArgs[0];
	int partsCount = getPartsCount(parentDimension);
	int begin = parentDimension.range.min + partId * size;
	int remaining = parentDimension.range.max - begin + 1;
	int intervalLength = (remaining >= size) ? size : remaining;

	if (paddings[0] > 0) {
		int frontPadding = getEffectiveFrontPadding(partId, parentDimension);
		begin -= frontPadding;
		intervalLength += frontPadding;
	}
	if (paddings[1] > 0) {
		intervalLength += getEffectiveRearPadding(partId, parentDimension);
	}

	Dimension partDimension;
	partDimension.range.min = begin;
	partDimension.range.max = begin + intervalLength - 1;
	partDimension.setLength();
	return partDimension;
}

int BlockSizeConfig::getEffectiveFrontPadding(int partId, Dimension parentDimension) {
	if (paddings[0] == 0) return 0;
	int size = partitionArgs[0];
	int partsCount = getPartsCount(parentDimension);
	int begin = parentDimension.range.min + partId * size;
	int paddedBegin = std::max(parentDimension.range.min, begin - paddings[0]);
	return begin - paddedBegin;
}

int BlockSizeConfig::getEffectiveRearPadding(int partId, Dimension parentDimension) {
	if (paddings[1] == 0) return 0;
	int size = partitionArgs[0];
	int partsCount = getPartsCount(parentDimension);
	int begin = parentDimension.range.min + partId * size;
	int remaining = parentDimension.range.max - begin + 1;
	int length = (remaining >= size) ? size : remaining;
	int end = begin + length - 1;
	int paddedEnd = std::min(parentDimension.range.max, end + paddings[1]);
	return paddedEnd - end;
}

PartitionInstr *BlockSizeConfig::getPartitionInstr() {
	int size = partitionArgs[0];
	BlockSizeInstr *instr = new BlockSizeInstr(size);
	Assert(instr != NULL);
	instr->setPadding(paddings[0], paddings[1]);
	return instr;
}

//----------------------------------------------------------- Block Count Config ----------------------------------------------------------/

int BlockCountConfig::getPartsCount(Dimension parentDimension) {
	int count = partitionArgs[0];
	int length = parentDimension.length;
        return std::max(1, std::min(count, length));
}

Dimension BlockCountConfig::getPartDimension(int partId, Dimension parentDimension) {
	
	int count = getPartsCount(parentDimension);
	int size = parentDimension.length / count;
	int begin = partId * size;
	int length = (partId < count - 1) ? size : parentDimension.range.max - begin + 1;

	if (paddings[0] > 0) {
		int frontPadding = getEffectiveFrontPadding(partId, parentDimension);
		begin -= frontPadding;
		length += frontPadding;
	}
	if (paddings[1] > 0) {
		length += getEffectiveRearPadding(partId, parentDimension);
	}

	Dimension partDimension;
	partDimension.range.min = begin;
	partDimension.range.max = begin + length - 1;
	partDimension.setLength();
	return partDimension;
}

int BlockCountConfig::getEffectiveFrontPadding(int partId, Dimension parentDimension) {
	if (paddings[0] == 0) return 0;
	int count = getPartsCount(parentDimension);
	int size = parentDimension.length / count;
	int begin = partId * size;
	int paddedBegin = std::max(parentDimension.range.min, begin - paddings[0]);
	return begin - paddedBegin;
}
               
int BlockCountConfig::getEffectiveRearPadding(int partId, Dimension parentDimension) {
	if (paddings[1] == 0) return 0;
	int count = getPartsCount(parentDimension);
	int size = parentDimension.length / count;
	int begin = partId * size;
	int length = (partId < count - 1) ? size : parentDimension.range.max - begin + 1;
	int end = begin + length - 1;
	int paddedEnd = std::min(parentDimension.range.max, end + paddings[1]);
	return paddedEnd - end;
}

PartitionInstr *BlockCountConfig::getPartitionInstr() {
	int count = partitionArgs[0];
	BlockCountInstr *instr = new BlockCountInstr(count);
	Assert(instr != NULL);
	instr->setPadding(paddings[0], paddings[1]);
	return instr;
}

//-------------------------------------------------------------- Stride Config ------------------------------------------------------------/

int StrideConfig::getPartsCount(Dimension parentDimension) {
	int length = parentDimension.length;
        return std::max(1, std::min(ppuCount, length));
}

Dimension StrideConfig::getPartDimension(int partId, Dimension parentDimension) {
	int partsCount = getPartsCount(parentDimension);
	int length = parentDimension.length;
	int perStrideEntries = length / partsCount;
        int myEntries = perStrideEntries;
        int remainder = length % partsCount;
        int extra = 0;
        if (remainder > 0) extra = remainder;
        if (remainder > partId) {
                myEntries++;
                extra = partId;
        }
        Dimension partDimension;
        partDimension.range.min = parentDimension.range.min + perStrideEntries * partId + extra;
        partDimension.range.max = partDimension.range.min + myEntries - 1;
        partDimension.setLength();
	return partDimension.getNormalizedDimension();
}

int StrideConfig::getOriginalIndex(int partIndex, int position, List<int> *partIdList,        
		List<int> *partCountList,
		List<Dimension*> *partDimensionList) {

	int partId = partIdList->Nth(position);
	int partCount = partCountList->Nth(position);
	int originalIndex = partId + partIndex * partCount;
	
	if (position > 0) {
		Dimension *parentDimension = partDimensionList->Nth(position - 1);
		originalIndex += parentDimension->range.min;
	} else {
		originalIndex += dataDimension.range.min;
	}
	
	return DimPartitionConfig::getOriginalIndex(originalIndex, position, partIdList, 
			partCountList, partDimensionList);
}

//----------------------------------------------------------- Block Stride Config ---------------------------------------------------------/

int BlockStrideConfig::getPartsCount(Dimension parentDimension) {
	int blockSize = partitionArgs[0];
	int length = parentDimension.length;
        int strides = length / blockSize;
        return std::max(1, std::min(strides, ppuCount));
}

Dimension BlockStrideConfig::getPartDimension(int partId, Dimension parentDimension) {

	int partsCount = getPartsCount(parentDimension);
	int blockSize = partitionArgs[0];
        int strideLength = blockSize * partsCount;
        int strideCount = parentDimension.length / strideLength;
        int myEntries = strideCount * blockSize;
        
	int partialStrideElements = parentDimension.length % strideLength;
        int blockCount = partialStrideElements / blockSize;
        int extraEntriesBefore = partialStrideElements;

        // if extra entries fill up a complete new block in the stride of the current LPU then its number of 
	// entries should increase by the size parameter and extra preceding entries should equal to 
	// block_size * preceding strides count
        if (blockCount > partId) {
                myEntries += blockSize;
                extraEntriesBefore = partId * blockSize;
        // If the extra entries does not fill a complete block for the current one then it should have whatever 
	// remains after filling up preceding blocks
        } else if (blockCount == partId) {
                myEntries += extraEntriesBefore - partId * blockSize;
                extraEntriesBefore = partId * blockSize;
        }
        
	Dimension partDimension;
        partDimension.range.min = parentDimension.range.min 
			+ partId * strideCount * strideLength + extraEntriesBefore;
        partDimension.range.max = partDimension.range.min + myEntries - 1;
        partDimension.setLength();
        return partDimension.getNormalizedDimension();
}

int BlockStrideConfig::getOriginalIndex(int partIndex, int position, List<int> *partIdList,
		List<int> *partCountList,
		List<Dimension*> *partDimensionList) {
	
	int partId = partIdList->Nth(position);
	int partCount = partCountList->Nth(position);
	int blockSize = partitionArgs[0];
	int originalIndex = ((partIndex / blockSize) * partCount + partId) * blockSize 
			+ partIndex % blockSize;
	
	if (position > 0) {
		Dimension *parentDimension = partDimensionList->Nth(position - 1);
		originalIndex += parentDimension->range.min;
	} else {
		originalIndex += dataDimension.range.min;
	}
	
	return DimPartitionConfig::getOriginalIndex(originalIndex, position, partIdList, 
			partCountList, partDimensionList);
}

PartitionInstr *BlockStrideConfig::getPartitionInstr() {
	int blockSize = partitionArgs[0];
	PartitionInstr *instr = new BlockStrideInstr(ppuCount, blockSize);
	Assert(instr != NULL);
	return instr; 
}

//---------------------------------------------------------- Data Partition Config --------------------------------------------------------/

DataPartitionConfig::DataPartitionConfig(int dimensionCount, List<DimPartitionConfig*> *dimensionConfigs) {
	this->dimensionCount = dimensionCount;
	this->dimensionConfigs = dimensionConfigs;
	this->parent = NULL;
	this->dimensionOrder = NULL;
}

void DataPartitionConfig::setParent(DataPartitionConfig *parent, int parentJump) { 
	this->parent = parent; 
	for (int i = 0; i < dimensionCount; i++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		DimPartitionConfig *parentConfig = parent->dimensionConfigs->Nth(i);
		dimConfig->setParentConfig(parentConfig);
	}
	this->parentJump = parentJump;
}

void DataPartitionConfig::configureDimensionOrder() {
	
	this->dimensionOrder = new std::vector<DimConfig>;
	Assert(this->dimensionOrder != NULL);
	if (parent != NULL) {
		std::vector<DimConfig> *parentOrder = parent->getDimensionOrder();
		for (int i = 0; i < parentOrder->size(); i++) {
			this->dimensionOrder->push_back(parentOrder->at(i));
		}
	}

	// ordering of the dimensions in the part-hierarchy-container-tree should follow their alignment order
	// with the LPS dimensions to make part searching take less steps 
	int level = getPartIdLevels() - 1;
	std::vector<int> lpsOrdering;
	std::vector<int> configOrdering;
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
		int alignment = dimConfig->getLpsAlignment();
		if (alignment != -1) {
			int position = binsearch::locatePointOfInsert(lpsOrdering, alignment);
			configOrdering.insert(configOrdering.begin() + position, d);
			lpsOrdering.insert(lpsOrdering.begin() + position, alignment);
		}
	}
	
	for (int i = 0; i < configOrdering.size(); i++) {
		this->dimensionOrder->push_back(DimConfig(level, configOrdering[i]));
	}

	// ordering of the replicated/un-partitioned dimensions does not matter; so just append them at the end
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
		int alignment = dimConfig->getLpsAlignment();
		if (alignment == -1) {
			this->dimensionOrder->push_back(DimConfig(level, d));
		}
	}
}
        
std::vector<DimConfig> *DataPartitionConfig::getDimensionOrder() {
	if (dimensionOrder == NULL) {
		configureDimensionOrder();
	}
	return dimensionOrder;
}

PartMetadata *DataPartitionConfig::generatePartMetadata(List<int*> *partIdList) {
	
	Assert(dimensionCount > 0);
	Dimension *partDimensions = new Dimension[dimensionCount];
	Assert(partDimensions != NULL);

	int *padding = new int[dimensionCount * 2];
	Assert(padding != NULL);
	for (int i = 0; i < dimensionCount; i++) {

		List<int> *dimIdList = new List<int>;
		Assert(dimIdList != NULL);
		for (int j = 0; j < partIdList->NumElements(); j++) {
			int *partId = partIdList->Nth(j);
			dimIdList->Append(partId[i]);
		}

		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		DimensionMetadata *dimMetadata = dimConfig->generateDimMetadata(dimIdList);
		partDimensions[i] = dimMetadata->partDimension;
		padding[2 * i] = dimMetadata->paddings[0];
		padding[2 * i + 1] = dimMetadata->paddings[1];

		delete dimIdList;
		delete dimMetadata;
	}

	PartMetadata *metadata = new PartMetadata(dimensionCount, partIdList, partDimensions, padding);
	Assert(metadata != NULL);
	return metadata;
}

List<int*> *DataPartitionConfig::generatePartId(List<int*> *lpuIds) {
	List<int*> *partId = new List<int*>;
	Assert(partId != NULL);
	int position = lpuIds->NumElements() - 1;
	generatePartId(lpuIds, position, partId);
	return partId;
}

void DataPartitionConfig::generatePartId(List<int*> *lpuIds, List<int*> *partId) {
	int partIdSteps = partId->NumElements();
	int position = lpuIds->NumElements() - 1;
	generatePartId(lpuIds, position, partId, true, partIdSteps - 1);
}

int DataPartitionConfig::getPartIdLevels() {
	int levels = 1;
	DataPartitionConfig *lastLink = this;
	while ((lastLink->parent != NULL)) {
		levels++;
		lastLink = lastLink->parent;		
	}
	return levels;
}

List<int*> *DataPartitionConfig::generatePartIdTemplate() {
	int levels = getPartIdLevels();
	List<int*> *templateId = new List<int*>(levels);
	Assert(templateId != NULL && dimensionCount > 0);
	for (int i = 0; i < levels; i++) {
		templateId->Append(new int[dimensionCount]);
	}
	return templateId;
}

List<int*> *DataPartitionConfig::generateSuperPartId(List<int*> *lpuIds, int backsteps, List<int*> *idTemplate) {
	
	List<int*> *partId = NULL;
	bool updateOnTemplate = false;
	if (idTemplate != NULL) {
		partId = idTemplate;
		updateOnTemplate = true;
	} else {
		partId = new List<int*>;
	}
	Assert(partId != NULL);

	int position = lpuIds->NumElements() - 1;
	int steps = 0;
	DataPartitionConfig *lastConfig = this;
	while (steps < backsteps) {
		position -= lastConfig->parentJump;
		lastConfig = lastConfig->parent;
		steps++;
	}
	Assert(position >= 0);

	if (updateOnTemplate) {
		int partIdSteps = idTemplate->NumElements();
		lastConfig->generatePartId(lpuIds, position, partId, true, partIdSteps - 1);
	} else {
		lastConfig->generatePartId(lpuIds, position, partId);
	}
	return partId;
}

DataPartsList *DataPartitionConfig::generatePartList(int epochCount) {
	Assert(dimensionCount > 0);
	Dimension *dataDimensions = new Dimension[dimensionCount];
	Assert(dataDimensions != NULL);
	bool hasPadding = false;
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
		dataDimensions[d] = dimConfig->getDataDimension();
		hasPadding = hasPadding || dimConfig->hasPadding();
	}
	ListMetadata *listMetadata = new ListMetadata(dimensionCount, dataDimensions);
	Assert(listMetadata != NULL);
	listMetadata->setPadding(hasPadding);
	DataPartsList *dataPartsList = new DataPartsList(listMetadata, epochCount);
	Assert(dataPartsList != NULL);
	return dataPartsList;
}

void DataPartitionConfig::generatePartId(List<int*> *lpuIds, int position, 
		List<int*> *partId, 
		bool updateExistingPartId, int updatePoint) {
	if (parent != NULL) {
		parent->generatePartId(lpuIds, position - parentJump, partId, updateExistingPartId, updatePoint - 1);
	}
	int *lpuId = lpuIds->Nth(position);
	int *partIdForLpu = NULL;
	if (!updateExistingPartId) {
		partIdForLpu = new int[dimensionCount];
		Assert(partIdForLpu != NULL && dimensionCount > 0);
	} else {
		partIdForLpu = partId->Nth(updatePoint);
	}
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimensionConfig = dimensionConfigs->Nth(d);
		partIdForLpu[d] = dimensionConfig->pickPartId(lpuId);
	}
	if (!updateExistingPartId) {
		partId->Append(partIdForLpu);
	}
}

int DataPartitionConfig::getPartsCountAlongDimension(int dimensionNo, Dimension *parentDimension) {
	DimPartitionConfig *dimConfig = dimensionConfigs->Nth(dimensionNo);
	if (parentDimension == NULL) {
		Dimension dataDimension = dimConfig->getDataDimension();
		return dimConfig->getPartsCount(dataDimension);
	} else {
		return dimConfig->getPartsCount(*parentDimension);
	}
}

void DataPartitionConfig::updatePartDimensionInfo(List<int*> *partIdList, int *lpuCounts, PartDimension *partDimension) {
	for (int i = 0; i < dimensionCount; i++) {
		List<int> *dimIdList = new List<int>;
		Assert(dimIdList != NULL);
		for (int j = 0; j < partIdList->NumElements(); j++) {
			int *partId = partIdList->Nth(i);
			dimIdList->Append(partId[i]);
		}
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		Dimension dimension = dimConfig->getPartDimension(dimIdList);
		partDimension[i].partition = dimension;
		partDimension[i].count = dimConfig->pickPartCount(lpuCounts);
	}
	int *lastPartId = partIdList->Nth(partIdList->NumElements() - 1);
	for (int i = 0; i < dimensionCount; i++) {
		partDimension[i].index = lastPartId[i];
	}	
}

void DataPartitionConfig::updatePartDimensionInfo(int *lpuId, 
		int *lpuCounts, PartDimension *partDims, PartDimension *parentPartDims) {	
	
	for (int i = 0; i < dimensionCount; i++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		Dimension parentDim = parentPartDims[i].partition;
		int partIdInDim = dimConfig->pickPartId(lpuId);
		partDims[i].partition = dimConfig->getPartDimension(partIdInDim, parentDim);
		partDims[i].parent = &(parentPartDims[i]);
		partDims[i].count = dimConfig->pickPartCount(lpuCounts);
		partDims[i].index = partIdInDim;
	}
}

ListMetadata *DataPartitionConfig::generatePartListMetadata(List<List<int*>*> *partIds) {
	List<PartMetadata*> *partMetadataList = new List<PartMetadata*>;
	Assert(partMetadataList != NULL);
	for (int i = 0; i < partIds->NumElements(); i++) {
		List<int*> *partIdList = partIds->Nth(i);
		partMetadataList->Append(generatePartMetadata(partIdList));
	}
	Dimension *dataDimensions = new Dimension[dimensionCount];
	Assert(dataDimensions != NULL && dimensionCount > 0);
	bool hasPadding = false;
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
		dataDimensions[d] = dimConfig->getDataDimension();
		hasPadding = hasPadding || dimConfig->hasPadding();
	}
	ListMetadata *listMetadata = new ListMetadata(dimensionCount, dataDimensions);
	Assert(listMetadata != NULL);
	listMetadata->setPadding(hasPadding);
	return listMetadata;
}

DataItemConfig *DataPartitionConfig::generateStateFulVersion() {
	
	DataPartitionConfig *current = this;
	int levelCount = 1;
	while (current->parent != NULL) {
		levelCount++;
		current = current->parent;
	}
	DataItemConfig *dataItemConfig = new DataItemConfig(dimensionCount, levelCount);
	Assert(dataItemConfig != NULL);
	
	for (int i = 0; i < dimensionCount; i++) {
		Dimension dataDimension = dimensionConfigs->Nth(i)->getDataDimension();
		dataItemConfig->setDimension(i, dataDimension);
	}

	std::stack<DataPartitionConfig*> configStack;
	current = this;
	while (current != NULL) {
		configStack.push(current);
		current = current->parent;
	}

	int level = 0;
	int entryCovered = 0;
	while (!configStack.empty()) {
		
		current = configStack.top();
		configStack.pop();
		dataItemConfig->setLpsIdOfLevel(level, current->getLpsId());

		for (int i = 0; i < dimensionCount; i++) {
			PartitionInstr *instr = current->getDimensionConfig(i)->getPartitionInstr();
			dataItemConfig->setPartitionInstr(level, i, instr);
		}
		
		std::vector<DimConfig> *dimOrder = current->getDimensionOrder();
		int size = dimOrder->size();
		std::vector<DimConfig> orderInCurrentLevel;
		orderInCurrentLevel.insert(orderInCurrentLevel.begin(), 
				dimOrder->begin() + entryCovered, dimOrder->begin() + size);
		for (int i = 0; i < orderInCurrentLevel.size(); i++) {
			DimConfig dimConfig = orderInCurrentLevel.at(i);
			int dimension = dimConfig.getDimNo();
			dataItemConfig->getInstruction(level, dimension)->setPriorityOrder(i);
		}

		entryCovered += orderInCurrentLevel.size(); 	
		level++;
	}
	
	dataItemConfig->updateParentLinksOnPartitionConfigs();
	return dataItemConfig;
}
