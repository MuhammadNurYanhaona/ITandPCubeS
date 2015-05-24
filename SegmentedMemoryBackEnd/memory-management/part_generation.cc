#include "part_generation.h"
#include "allocation.h"
#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../codegen/structure.h"
#include <algorithm>

//------------------------------------------------------------ Block Size Config ----------------------------------------------------------/

int BlockSizeConfig::getPartsCount() {
	int size = partitionArgs[0];
	int dimLength = dimension.length;
	return (dimLength + size - 1) / size;
}

LineInterval *BlockSizeConfig::getCoreInterval(int partId) {
	int size = partitionArgs[0];
	int partsCount = getPartsCount();
	int begin = dimension.range.min + partId * size;
	int remaining = dimension.range.max - begin + 1;
	int intervalLength = (remaining >= size) ? size : remaining;
	LineInterval *interval = new LineInterval(begin, intervalLength, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

LineInterval *BlockSizeConfig::getInterval(int partId) {
	LineInterval *coreInterval = getCoreInterval(partId);
	int begin = coreInterval->getBegin();
	int length = coreInterval->getLength();
	if (paddings[0] > 0) {
		int frontPadding = getEffectiveFrontPadding(partId);
		begin -= frontPadding;
		length += frontPadding;
	}
	if (paddings[1] > 0) {
		length += getEffectiveRearPadding(partId);
	}
	LineInterval *interval = new LineInterval(begin, length, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

Dimension BlockSizeConfig::getPartDimension(int partId) {
	LineInterval *interval = getInterval(partId);
	int begin = interval->getBegin();
	int end = begin + interval->getLength() - 1;
	Dimension partDimension;
	partDimension.range.min = begin;
	partDimension.range.max = end;
	partDimension.setLength();
	return partDimension;
}

int BlockSizeConfig::getEffectiveFrontPadding(int partId) {
	if (paddings[0] == 0) return 0;
	int size = partitionArgs[0];
	int partsCount = getPartsCount();
	int begin = dimension.range.min + partId * size;
	int paddedBegin = std::max(dimension.range.min, begin - paddings[0]);
	return begin - paddedBegin;
}

int BlockSizeConfig::getEffectiveRearPadding(int partId) {
	if (paddings[1] == 0) return 0;
	int size = partitionArgs[0];
	int partsCount = getPartsCount();
	int begin = dimension.range.min + partId * size;
	int remaining = dimension.range.max - begin + 1;
	int length = (remaining >= size) ? size : remaining;
	int end = begin + length - 1;
	int paddedEnd = std::min(dimension.range.max, end + paddings[1]);
	return paddedEnd - end;
}

//----------------------------------------------------------- Block Count Config ----------------------------------------------------------/

int BlockCountConfig::getPartsCount() {
	int count = partitionArgs[0];
	int length = dimension.length;
        return std::max(1, std::min(count, length));
}
        
LineInterval *BlockCountConfig::getCoreInterval(int partId) {
	int count = getPartsCount();
	int size = dimension.length / count;
	int begin = partId * size;
	int length = (partId < count - 1) ? size : dimension.range.max - begin + 1;
	LineInterval *interval = new LineInterval(begin, length, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

LineInterval *BlockCountConfig::getInterval(int partId) {
	LineInterval *coreInterval = getCoreInterval(partId);
	int begin = coreInterval->getBegin();
	int length = coreInterval->getLength();
	if (paddings[0] > 0) {
		int frontPadding = getEffectiveFrontPadding(partId);
		begin -= frontPadding;
		length += frontPadding;
	}
	if (paddings[1] > 0) {
		length += getEffectiveRearPadding(partId);
	}
	LineInterval *interval = new LineInterval(begin, length, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

Dimension BlockCountConfig::getPartDimension(int partId) {
	LineInterval *interval = getInterval(partId);
	int begin = interval->getBegin();
	int end = begin + interval->getLength() - 1;
	Dimension partDimension;
	partDimension.range.min = begin;
	partDimension.range.max = end;
	partDimension.setLength();
	return partDimension;
}

int BlockCountConfig::getEffectiveFrontPadding(int partId) {
	if (paddings[0] == 0) return 0;
	int count = getPartsCount();
	int size = dimension.length / count;
	int begin = partId * size;
	int paddedBegin = std::max(dimension.range.min, begin - paddings[0]);
	return begin - paddedBegin;
}
               
int BlockCountConfig::getEffectiveRearPadding(int partId) {
	if (paddings[1] == 0) return 0;
	int count = getPartsCount();
	int size = dimension.length / count;
	int begin = partId * size;
	int length = (partId < count - 1) ? size : dimension.range.max - begin + 1;
	int end = begin + length - 1;
	int paddedEnd = std::min(dimension.range.max, end + paddings[1]);
	return paddedEnd - end;
}

//-------------------------------------------------------------- Stride Config ------------------------------------------------------------/

int StrideConfig::getPartsCount() {
	int length = dimension.length;
        return std::max(1, std::min(ppuCount, length));
}

LineInterval *StrideConfig::getCoreInterval(int partId) {
	int partsCount = getPartsCount();
	int length = dimension.length;
	int strides = length /partsCount;
	int remaining = length % partsCount;
	if (remaining > partId) strides++;
	int begin = dimension.range.min + partId;
	LineInterval *interval = new LineInterval(begin, 1, strides, partsCount);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

LineInterval *StrideConfig::getXformedCoreInterval(int partId) {
	Dimension partDimension = getPartDimension(partId);
	int begin = partDimension.range.min;
	int length = partDimension.length;
	LineInterval *interval = new LineInterval(begin, length, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

Dimension StrideConfig::getPartDimension(int partId) {
	int partsCount = getPartsCount();
	int length = dimension.length;
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
        partDimension.range.min = dimension.range.min + perStrideEntries * partId + extra;
        partDimension.range.max = partDimension.range.min + myEntries - 1;
        partDimension.setLength();
	return partDimension;
}

//----------------------------------------------------------- Block Stride Config ---------------------------------------------------------/

int BlockStrideConfig::getPartsCount() {
	int blockSize = partitionArgs[0];
	int length = dimension.length;
        int strides = length / blockSize;
        return std::max(1, std::min(strides, ppuCount));
}

LineInterval *BlockStrideConfig::getCoreInterval(int partId) {
	
	int partsCount = getPartsCount();
	int blockSize = partitionArgs[0];
        int strideLength = blockSize * partsCount;
        int strideCount = dimension.length / strideLength;

	// the stride count for the current part may be increased by one if dimension is not divisible
	// by the strideLength
	int partialStrideElements = dimension.length % strideLength;
        int extraBlockCount = partialStrideElements / blockSize;
	if (extraBlockCount > partId || 
		(extraBlockCount == partId 
		&& partialStrideElements % blockSize != 0)) strideCount++;

	int begin = blockSize * partId;
	int length = blockSize;
	int count = strideCount;
	int gap = strideLength;
	LineInterval *interval = new LineInterval(begin, length, count, gap);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

LineInterval *BlockStrideConfig::getXformedCoreInterval(int partId) {
	Dimension partDimension = getPartDimension(partId);
	int begin = partDimension.range.min;
	int length = partDimension.length;
	LineInterval *interval = new LineInterval(begin, length, 1, 0);
	interval->setLine(new Line(dimension.range.min, dimension.range.max));
	return interval;	
}

Dimension BlockStrideConfig::getPartDimension(int partId) {

	int partsCount = getPartsCount();
	int blockSize = partitionArgs[0];
        int strideLength = blockSize * partsCount;
        int strideCount = dimension.length / strideLength;
        int myEntries = strideCount * blockSize;
        
	int partialStrideElements = dimension.length % strideLength;
        int blockCount = partialStrideElements / blockSize;
        int extraEntriesBefore = partialStrideElements;

        // if extra entries fill up a complete new block in the stride of the current LPU then its number of 
	// entries should increase by the size parameter and extra preceeding entries should equal to 
	// block_size * preceeding strides count
        if (blockCount > partId) {
                myEntries += blockSize;
                extraEntriesBefore = partId * blockSize;
        // If the extra entries does not fill a complete block for the current one then it should have whatever 
	// remains after filling up preceeding blocks
        } else if (blockCount == partId) {
                myEntries += extraEntriesBefore - partId * blockSize;
                extraEntriesBefore = partId * blockSize;
        }
        
	Dimension partDimension;
        partDimension.range.min = partId * strideCount * strideLength + extraEntriesBefore;
        partDimension.range.max = partDimension.range.min + myEntries - 1;
        partDimension.setLength();
        return partDimension;
}

//---------------------------------------------------------- Data Partition Config --------------------------------------------------------/

PartMetadata *DataPartitionConfig::generatePartMetadata(int *partId) {
	
	Dimension *partDimensions = new Dimension[dimensionCount];
	List<LineInterval*> *linearIntervals = new List<LineInterval*>;
	List<LineInterval*> *paddedIntervals = new List<LineInterval*>;
	int *padding = new int[dimensionCount * 2];

	for (int i = 0; i < dimensionCount; i++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(i);
		int dimId = partId[i];
		linearIntervals->Append(dimConfig->getCoreInterval(dimId));
		paddedIntervals->Append(dimConfig->getInterval(dimId));
		partDimensions[i] = dimConfig->getPartDimension(dimId);
		padding[2 * i] = dimConfig->getEffectiveFrontPadding(dimId);
		padding[2 * i + 1] = dimConfig->getEffectiveRearPadding(dimId);
	}

	HyperplaneInterval *coreInterval = new HyperplaneInterval(
			dimensionCount, linearIntervals);
	HyperplaneInterval *paddedInterval = new HyperplaneInterval(
			dimensionCount, paddedIntervals);
	PartMetadata *metadata = new PartMetadata(dimensionCount, partId, partDimensions, padding);
	metadata->setIntervals(coreInterval, paddedInterval);
	return metadata;
}

int *DataPartitionConfig::getMultidimensionalLpuId(int lpsDimensions, int *lpuCount, int linearId) {
	int *id = new int[lpsDimensions];
	int subId = linearId;
	for (int i = 0; i < lpsDimensions; i++) {
		int denominator = 1;
		for (int j = i + 1; j < lpsDimensions; j++) {
			denominator *= lpuCount[j];
		}
		id[i] = subId / denominator;
		subId = subId % denominator;
	}
	return id;
}

List<int*> *DataPartitionConfig::getLpuIdsFromRange(int lpsDimensions, 
		int currentDimension, Range *localLpus) {
	
	Range currentRange = localLpus[currentDimension];
	// if this is not the last dimension of the lps then first do a recursion then update the list of ids
	// retrieved from the lower dimensions
	if (currentDimension < lpsDimensions - 1) {
		// recursive call for next lower dimension
		List<int*> *lowerIdList = getLpuIdsFromRange(lpsDimensions, currentDimension + 1, localLpus);
		int lowerIdDimensions = lpsDimensions - (currentDimension + 1);
		// generate a higher dimensional id including the values for current dimension on the result of
		// the recursion	
		List<int*> *updatedIdList = new List<int*>;
		for (int i = currentRange.min; i <= currentRange.max; i++) {
			int currDimId = i;
			for (int j = 0; j < lowerIdList->NumElements(); j++) {
				int *nextId = new int[lowerIdDimensions + 1];
				int *lowerId = lowerIdList->Nth(j);
				for (int d = 0; d < lowerIdDimensions; d++) {
					nextId[d + 1] = lowerId[d];
				}
				nextId[0] = currDimId;
				updatedIdList->Append(nextId); 
			}
		}
		// delete all ids from the previous list
		while (lowerIdList->NumElements() > 0) {
			int *lowerId = lowerIdList->Nth(0);
			lowerIdList->RemoveAt(0);
			delete[] lowerId;
		}
		// delete the previous list and return the updated list
		delete lowerIdList;
		return updatedIdList;
	// if this is the last LPS dimension then create ids for allocated entries directly from the last range
	} else {
		List<int*> *idList = new List<int*>;
		for (int i = currentRange.min; i <= currentRange.max; i++) {
			int *id = new int[1];
			id[0] = i;
			idList->Append(id);
		}
		return idList;	
	}
}

List<int*> *DataPartitionConfig::getLocalPartIds(int lpsDimensions, int *lpuCount, Range localRange) {
	List<int*> *lpuIdList = new List<int*>;
	for (int i = localRange.min; i <= localRange.max; i++) {
		int *lpuId = getMultidimensionalLpuId(lpsDimensions, lpuCount, i);
		lpuIdList->Append(lpuId);
	}
	List<int*> *partIdList = getLocalPartIds(lpuIdList);
	while (lpuIdList->NumElements() > 0) {
		int *lpuId = lpuIdList->Nth(0);
		lpuIdList->RemoveAt(0);
		delete[] lpuId;
	}
	delete lpuIdList;
	return partIdList;
}

List<int*> *DataPartitionConfig::getLocalPartIds(List<int*> *localLpuIds) {
	List<int*> *uniqueParts = new List<int*>;
	for (int i = 0; i < localLpuIds->NumElements(); i++) {
		int *lpuId = localLpuIds->Nth(i);
		int *partId = new int[dimensionCount];
		for (int d = 0; d < dimensionCount; d++) {
			DimPartitionConfig *dimensionConfig = dimensionConfigs->Nth(d);
			partId[d] = dimensionConfig->pickPartId(lpuId);
		}
		// check if the part has already been added in the list before
		bool partAlreadyAdded = false;
		for (int j = 0; j < uniqueParts->NumElements(); j++) {
			int *existingId = uniqueParts->Nth(j);
			bool match = true;
			for (int d = 0; d < dimensionCount; d++) {
				if (partId[d] != existingId[d]) {
					match = false; 
					break;
				}
			}
			if (match) {
				partAlreadyAdded = true;
				break;
			}
		}
		// if the part is redundant then delete it otherwise add it in the list
		if (partAlreadyAdded) delete[] partId;
		else uniqueParts->Append(partId);
	}
	return uniqueParts;
}

int *DataPartitionConfig::generatePartId(int *lpuId) {
	int *partId = new int[dimensionCount];
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimensionConfig = dimensionConfigs->Nth(d);
		partId[d] = dimensionConfig->pickPartId(lpuId);
	}
	return partId;
}

ListMetadata *DataPartitionConfig::generatePartsMetadata(List<int*> *partIds) {
	List<PartMetadata*> *partMetadataList = new List<PartMetadata*>;
	for (int i = 0; i < partIds->NumElements(); i++) {
		int *partId = partIds->Nth(i);
		partMetadataList->Append(generatePartMetadata(partId));
	}
	Dimension *dataDimensions = new Dimension[dimensionCount];
	bool hasPadding = false;
	for (int d = 0; d < dimensionCount; d++) {
		DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
		dataDimensions[d] = dimConfig->getDataDimension();
		hasPadding = hasPadding || dimConfig->hasPadding();
	}
	ListMetadata *listMetadata = new ListMetadata(dimensionCount, dataDimensions);
	listMetadata->setPadding(hasPadding);
	listMetadata->generateIntervalSpec(partMetadataList);
	return listMetadata;
}
