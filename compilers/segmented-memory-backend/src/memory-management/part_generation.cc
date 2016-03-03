#include "part_generation.h"
#include "allocation.h"
#include "part_tracking.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"
#include "../runtime/structure.h"
#include "../partition-lib/partition.h"
#include "../communication/part_config.h"

#include <stack>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string.h>
#include <cstdlib>

//--------------------------------------------------------- Part Interval Pattern ---------------------------------------------------------/

PartIntervalPattern::PartIntervalPattern() {
	beginExpr = NULL;
	count = 0;
	period = 0;
	length = 0;
	overflow = 0;
	instances = 0;
}

PartIntervalPattern::PartIntervalPattern(int c, int p, int l, int o, int i) {
	beginExpr = NULL;
	count = c;
	period = p;
	length = l;
	overflow = o;
	instances = i;
}

bool PartIntervalPattern::isEqual(PartIntervalPattern *other) {
	return (strcmp(beginExpr, other->beginExpr) == 0)
			&& (count == other->count)
			&& (period == other->period)
			&& (length == other->length)
			&& (overflow == other->overflow)
			&& (instances == other->instances);
}

bool PartIntervalPattern::isEqualIgnoringInstanceCount(PartIntervalPattern *other) {
	return (strcmp(beginExpr, other->beginExpr) == 0)
			&& (count == other->count)
			&& (period == other->period)
			&& (length == other->length)
			&& (overflow == other->overflow);
}

int PartIntervalPattern::getPartLength() {
	return count * length + overflow;
}

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

List<PartIntervalPattern*> *DimPartitionConfig::getPartIntervalPatterns(Dimension origDimension) {
	List<PartIntervalPattern*> *list = new List<PartIntervalPattern*>;
	PartIntervalPattern *pattern = new PartIntervalPattern;
	pattern->count = 1;
	pattern->period = origDimension.getLength();
	pattern->length = pattern->period;
	pattern->instances = 1;
	std::ostringstream stream;
	stream << origDimension.range.min;
	pattern->beginExpr = strdup(stream.str().c_str());
	list->Append(pattern);
	return list;
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

bool BlockSizeConfig::isEqual(DimPartitionConfig *otherConfig) {
	BlockSizeConfig *other = dynamic_cast<BlockSizeConfig*>(otherConfig);
	if (other == NULL) return false;
	return (partitionArgs[0] == other->partitionArgs[0])
			&& (paddings[0] == other->paddings[0])
			&& (paddings[1] == other->paddings[1]);
}

List<PartIntervalPattern*> *BlockSizeConfig::getPartIntervalPatterns(Dimension origDimension) {
	
	int partsCount = getPartsCount(origDimension);
	if (partsCount == 1) return DimPartitionConfig::getPartIntervalPatterns(origDimension);
	
	int size = partitionArgs[0];
	List<PartIntervalPattern*> *patternList = new List<PartIntervalPattern*>;
	
	// separately create a pattern description for the first part as it might differ from the rest if there is a
	// non-zero front padding
	Dimension firstPartDim = getPartDimension(0, origDimension);
	PartIntervalPattern *pattern = new PartIntervalPattern(1, firstPartDim.length, firstPartDim.length, 0, 1);
	std::ostringstream stream;
	stream << "partId * " << size;
	pattern->beginExpr = strdup(stream.str().c_str());
	patternList->Append(pattern);

	// create a pattern for the rest of the parts	
	Dimension secondPartDim = getPartDimension(1, origDimension);
	int instances = partsCount - 1;
	pattern = new PartIntervalPattern(1, secondPartDim.length, secondPartDim.length, 0, instances);
	if (paddings[0] > 0) {
		stream << " - " << paddings[0];
	}
	pattern->beginExpr = strdup(stream.str().c_str());
	// only add the second pattern in the list if it differs from the first pattern
	if (pattern->isEqualIgnoringInstanceCount(patternList->Nth(0))) {
		patternList->Nth(0)->instances = partsCount;
		delete pattern;
	} else {
		patternList->Append(pattern);
	}

	// if the original dimension is not evenly divided by the partition instruction or there is a non-zero back 
	// padding then the last part's pattern should differ from that of the others
	if (partsCount > 2 && (origDimension.length % size != 0 || paddings[1] != 0)) {
		
		// first decrease the instance count from the recently added pattern
		int patternCount = patternList->NumElements();
		pattern = patternList->Nth(patternCount - 1);
		pattern->instances = pattern->instances - 1;

		// then add the exception pattern in the list
		Dimension lastPartDim = getPartDimension(partsCount - 1, origDimension);
		pattern = new PartIntervalPattern(1, lastPartDim.length, lastPartDim.length, 0, 1);
		pattern->beginExpr = strdup(stream.str().c_str());
		patternList->Append(pattern);	
	}
	return patternList;	
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
	int begin = parentDimension.range.min + partId * size;
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
	int begin = parentDimension.range.min + partId * size;
	int paddedBegin = std::max(parentDimension.range.min, begin - paddings[0]);
	return begin - paddedBegin;
}
               
int BlockCountConfig::getEffectiveRearPadding(int partId, Dimension parentDimension) {
	if (paddings[1] == 0) return 0;
	int count = getPartsCount(parentDimension);
	int size = parentDimension.length / count;
	int begin = parentDimension.range.min + partId * size;
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

bool BlockCountConfig::isEqual(DimPartitionConfig *otherConfig) {
	BlockCountConfig *other = dynamic_cast<BlockCountConfig*>(otherConfig);
	if (other == NULL) return false;
	return (partitionArgs[0] == other->partitionArgs[0])
			&& (paddings[0] == other->paddings[0])
			&& (paddings[1] == other->paddings[1]);
}

// the implementation of this function for the block-count-configuration matches almost exactly with the implementation 
// for the block-size-configuration
List<PartIntervalPattern*> *BlockCountConfig::getPartIntervalPatterns(Dimension origDimension) {
	
	int partsCount = getPartsCount(origDimension);
	if (partsCount == 1) return DimPartitionConfig::getPartIntervalPatterns(origDimension);

	// calculation of the parts' size is the place where the logics for block-count and block-size configuration
	// differs 
	int size = origDimension.length / partsCount;

	List<PartIntervalPattern*> *patternList = new List<PartIntervalPattern*>;
	
	// separately create a pattern description for the first part as it might differ from the rest if there is a
	// non-zero front padding
	Dimension firstPartDim = getPartDimension(0, origDimension);
	PartIntervalPattern *pattern = new PartIntervalPattern(1, firstPartDim.length, firstPartDim.length, 0, 1);
	std::ostringstream stream;
	stream << "partId * " << size;
	pattern->beginExpr = strdup(stream.str().c_str());
	patternList->Append(pattern);

	// create a pattern for the rest of the parts	
	Dimension secondPartDim = getPartDimension(1, origDimension);
	int instances = partsCount - 1;
	pattern = new PartIntervalPattern(1, secondPartDim.length, secondPartDim.length, 0, instances);
	if (paddings[0] > 0) {
		stream << " - " << paddings[0];
	}
	pattern->beginExpr = strdup(stream.str().c_str());
	// only add the second pattern in the list if it differs from the first pattern
	if (pattern->isEqualIgnoringInstanceCount(patternList->Nth(0))) {
		patternList->Nth(0)->instances = partsCount;
		delete pattern;
	} else {
		patternList->Append(pattern);
	}

	// if the original dimension is not evenly divided by the partition instruction or there is a non-zero back 
	// padding then the last part's pattern should differ from that of the others
	if (partsCount > 2 && (origDimension.length % size != 0 || paddings[1] != 0)) {
		
		// first decrease the instance count from the recently added pattern
		int patternCount = patternList->NumElements();
		pattern = patternList->Nth(patternCount - 1);
		pattern->instances = pattern->instances - 1;

		// then add the exception pattern in the list
		Dimension lastPartDim = getPartDimension(partsCount - 1, origDimension);
		pattern = new PartIntervalPattern(1, lastPartDim.length, lastPartDim.length, 0, 1);
		pattern->beginExpr = strdup(stream.str().c_str());
		patternList->Append(pattern);	
	}
	return patternList;	
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

bool StrideConfig::isEqual(DimPartitionConfig *otherConfig) {
	StrideConfig *other = dynamic_cast<StrideConfig*>(otherConfig);
	if (other == NULL) return false;
	return ppuCount == other->ppuCount;
}

List<PartIntervalPattern*> *StrideConfig::getPartIntervalPatterns(Dimension origDimension) {
		
	int partsCount = getPartsCount(origDimension);
	if (partsCount == 1) return DimPartitionConfig::getPartIntervalPatterns(origDimension);

	// determine if the number of stride steps and if the strides divide the original dimension evenly
	int steps = origDimension.length / partsCount;
	int remainder = origDimension.length % partsCount;

	List<PartIntervalPattern*> *patternList = new List<PartIntervalPattern*>;
	
	// first create a pattern assuming that there is no overflow, that is, the remainder is zero
	PartIntervalPattern *pattern = new PartIntervalPattern(steps, partsCount, 1, 0, partsCount);
	std::ostringstream stream;
	stream << "partId * 1";
	pattern->beginExpr = strdup(stream.str().c_str());

	// if there is a reminder than earlier parts should have an extra steps and we need to reduce the number of
	// instances we registered for the already included parts
	if (remainder > 0) {
		// reduce the instance count
		pattern->instances = partsCount - remainder;
		// create a new pattern and add that at the beginning
		pattern = new PartIntervalPattern(steps + 1, partsCount, 1, 0, remainder);
		pattern->beginExpr = strdup(stream.str().c_str());
		patternList->InsertAt(pattern, 0);
	}	

	return patternList;
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

bool BlockStrideConfig::isEqual(DimPartitionConfig *otherConfig) {
	BlockStrideConfig *other = dynamic_cast<BlockStrideConfig*>(otherConfig);
	if (other == NULL) return false;
	return (ppuCount == other->ppuCount) && (partitionArgs[0] == other->partitionArgs[0]);
}

List<PartIntervalPattern*> *BlockStrideConfig::getPartIntervalPatterns(Dimension origDimension) {
	
	int partsCount = getPartsCount(origDimension);
	if (partsCount == 1) return DimPartitionConfig::getPartIntervalPatterns(origDimension);
	
	int blockSize = partitionArgs[0];
	List<PartIntervalPattern*> *patternList = new List<PartIntervalPattern*>;
	
	// There are three possible patterns for a block stride configuration. In the general case, most parts will
	// have the same number of stride steps. In case the dimension is not partitioned evenly, however, some parts
	// may have an extra steps; and at most one part may have an overflow, i.e., a partial steps.
	int strideLength = blockSize * partsCount;
	int steps = origDimension.length / strideLength;
	int remainder = origDimension.length % strideLength;
	int partialBlocks = remainder / blockSize;
	int overflow = remainder % blockSize;

	// create a begin expression common to all parts
	std::ostringstream stream;
	stream << stream << "partid * " << blockSize;
	const char *beginExpr = strdup(stream.str().c_str());
	
	// create a pattern for the general case parts
	int count1 = partsCount - partialBlocks - ((overflow > 0) ? 1 : 0);
	if (count1 > 0) {
		PartIntervalPattern *pattern = new PartIntervalPattern(steps, strideLength, blockSize, 0, count1);
		pattern->beginExpr = beginExpr;
		patternList->Append(pattern);
	}

	// we are adding patterns in the reverse order here; so if there is an overflow block we add it next
	if (overflow > 0) {
		PartIntervalPattern *pattern = new PartIntervalPattern(steps, strideLength, blockSize, overflow, 1);
		pattern->beginExpr = beginExpr;
		patternList->InsertAt(pattern, 0);
	}

	// finally add the parts pattern that have one more extra step in the stride
	if (partialBlocks > 0) {
		PartIntervalPattern *pattern = new PartIntervalPattern(steps + 1, 
				strideLength, blockSize, 0, partialBlocks);
		pattern->beginExpr = beginExpr;
		patternList->InsertAt(pattern, 0);
	} 

	return patternList;
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

bool DataPartitionConfig::isEquivalent(DataPartitionConfig *other) {
	
	// first compare the root data-dimenions the two configurations are trying to divide into parts; if they are
	// different then the parts lists cannot be the same
	for (int i = 0; i < dimensionCount; i++) {
		Dimension myDim = dimensionConfigs->Nth(i)->getDataDimension();
		Dimension otherDim = other->dimensionConfigs->Nth(i)->getDataDimension();
		if (!myDim.isEqual(otherDim)) return false;
	}

	// then construct partition hierarchies for the two configurations
	List<std::vector<DimPartitionConfig*>*> *myHierarchy = new List<std::vector<DimPartitionConfig*>*>; 
	this->preparePartitionHierarchy(myHierarchy);
	List<std::vector<DimPartitionConfig*>*> *otherHierarchy = new List<std::vector<DimPartitionConfig*>*>;
	other->preparePartitionHierarchy(otherHierarchy);

	// if the two hierarchies have different lengths then, given the degenerarive instructions are removed, the
	// parts being generated by the two configurations are most likely different
	if (myHierarchy->NumElements() != otherHierarchy->NumElements()) return false;

 
	// determine if the number of times index reordering happens in the two hierarchies are different; if the are
	// then the resulting parts are certainly different
	// While we are calculating the number of reorderings along the tow hierarchy; we can also determine if one has
	// more partition instructions at any level, or different dimensions are partitionned at a level by the two
	// hierarchies. If that happens then too the parts are different
	for (int i = 0; i < myHierarchy->NumElements(); i++) {
		std::vector<DimPartitionConfig*> *configVector1 = myHierarchy->Nth(i);
		std::vector<DimPartitionConfig*> *configVector2 = otherHierarchy->Nth(i);
		for (int j = 0; j < dimensionCount; j++) {
			DimPartitionConfig *dimConfig1 = configVector1->at(j);
			DimPartitionConfig *dimConfig2 = configVector2->at(j);
			if ((dimConfig1 == NULL && dimConfig2 != NULL) 
					|| (dimConfig1 != NULL && dimConfig2 == NULL)
					|| (dimConfig1->doesReorderIndices() != dimConfig2->doesReorderIndices())) {
				return false;
			}
		}
	}

	// if the two hierarchies have the exact same partition instructions at all levels for all dimensions then the
	// parts are equivalent
	bool sameConfiguration = true;
	for (int i = 0; i < myHierarchy->NumElements(); i++) {
		std::vector<DimPartitionConfig*> *configVector1 = myHierarchy->Nth(i);
		std::vector<DimPartitionConfig*> *configVector2 = otherHierarchy->Nth(i);
		for (int j = 0; j < dimensionCount; j++) {
			DimPartitionConfig *dimConfig1 = configVector1->at(j);
			DimPartitionConfig *dimConfig2 = configVector2->at(j);
			if (dimConfig1 != NULL && !dimConfig1->isEqual(dimConfig2)) {
				sameConfiguration = false;
				break;
			}
		}
		if (!sameConfiguration) break;
	}
	if (sameConfiguration) return true;
	
	// There might be a chance that although the instructions in the two hierarchies do not exactly match, parts
	// generated by them will be the same nonetheless. Instead of generating all the parts and compare their metadata
	// to test for that case -- which is time and memory intensive -- we can recursively generate interval pattern 
	// expressions for the parts' dimensions then test for pattern equivalence along each dimension
	for (int i = 0; i < dimensionCount; i++) {
		Dimension dataDimension = dimensionConfigs->Nth(i)->getDataDimension();
		if(!generateSimilarParts(myHierarchy, otherHierarchy, i, 0, dataDimension)) return false;
	}    
	return true;	
}

void DataPartitionConfig::preparePartitionHierarchy(List<std::vector<DimPartitionConfig*>*> *hierarchy) {	
	
	std::vector<DimPartitionConfig*> *myVector = new std::vector<DimPartitionConfig*>;
	myVector->reserve(dimensionCount);
	
	bool activeLevel = false;
	for (int i = 0; i < dimensionCount; i++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		if (dimConfig->isDegenerativeCase()) {
			myVector->push_back(NULL);
		} else {
			myVector->push_back(dimConfig);
			activeLevel = true;
		}
	}
	if (activeLevel) {
		hierarchy->InsertAt(myVector, 0);
	} else {
		delete myVector;
	}
	if (parent != NULL) parent->preparePartitionHierarchy(hierarchy);
}

bool DataPartitionConfig::generateSimilarParts(List<std::vector<DimPartitionConfig*>*> *first,
		List<std::vector<DimPartitionConfig*>*> *second,
		int currentDimNo,
		int currentLevel,
		Dimension dimToDivide) {

	// Note that this function is called after deducing that the two configuration hierarchies have the same height.
	// So if the recursion reaches the end level of one hierarchy then there is nothing more to test and we can say
	// the parts will be similar  
	if (currentLevel == first->NumElements()) return true;

	// Previous validation ensures that if one hierarchy does not have any partition instruction at a particular 
	// level then the other hierarchy does not have any instruction at that level either. Thus, the call should be
	// forwarded to the next level
	DimPartitionConfig *firstConfig = first->Nth(currentLevel)->at(currentDimNo);
	if (firstConfig == NULL) {
		return generateSimilarParts(first, second, currentDimNo, currentLevel + 1, dimToDivide);
	}

	DimPartitionConfig *secondConfig = second->Nth(currentLevel)->at(currentDimNo);
	
	// Determine if the number of parts generated by the two configuration at the current level and dimension of
	// interest are different. If they are different then the parts are dissimilar.
	int partsCount1 = firstConfig->getPartsCount(dimToDivide);
	int partsCount2 = secondConfig->getPartsCount(dimToDivide);
	if (partsCount1 != partsCount2) return false;

	// retrieve the interval patterns of the parts generated by the two configurations
	List<PartIntervalPattern*> *firstPatternSet = firstConfig->getPartIntervalPatterns(dimToDivide);
	List<PartIntervalPattern*> *secondPatternSet = secondConfig->getPartIntervalPatterns(dimToDivide);

	// if the number of patterns in the two sets are different then the final parts are dissimilar
	if (firstPatternSet->NumElements() != secondPatternSet->NumElements()) return false;

	// compare the individual patterns for similarity
	for (int i = 0; i < firstPatternSet->NumElements(); i++) {
		PartIntervalPattern *firstPattern = firstPatternSet->Nth(i);
		PartIntervalPattern *secondPattern = secondPatternSet->Nth(i);
		if (!firstPattern->isEqual(secondPattern)) return false;
		
		// Even if the patterns are similar at the current level, when they are divided at the subsequent levels
		// by upcoming partition instructions, they can become different. Therefore, we need to roll the recursion
		// for each parts of the pattern generated here
		int partLength = firstPattern->getPartLength();
		// The begining of a part's dimension does not influence how it should be divided by partition instructions.
		// Thus, we can create a single dimension object of appropriate length for all parts
		Dimension nextDimension;
		nextDimension.setLength(partLength);
		bool match = generateSimilarParts(first, second, currentDimNo, currentLevel + 1, nextDimension);
		if (!match) return false; 
	}

	// if no differences are found in all descendent levels then the parts are similar
	return true;	
}
