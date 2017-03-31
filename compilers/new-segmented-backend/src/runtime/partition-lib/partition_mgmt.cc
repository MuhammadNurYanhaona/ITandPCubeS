#include "partition_mgmt.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/domain-obj/structure.h"

#include <iostream>
#include <algorithm>

/******************************** partitionCount functions ****************************************/

int block_size_partitionCount(Dimension d, int ppuCount, int size) {
        return (d.length + size - 1) / size;
}

int block_count_partitionCount(Dimension d, int ppuCount, int count) {
	int length = d.length;
        return std::max(1, std::min(count, length));
}

int stride_partitionCount(Dimension d, int ppuCount) {
	int length = d.length;
        return std::max(1, std::min(ppuCount, length));
}

int block_stride_partitionCount(Dimension d, int ppuCount, int size) {
	int length = d.length;
	int strides = length / size;
        return std::max(1, std::min(strides, ppuCount));
}

/*********************************** getRange functions *******************************************/

Dimension block_size_getRange(Dimension d, 
		int lpuCount, 
		int lpuId, 
		bool copyMode, 
		int size, 
		int frontPadding, 
		int backPadding) {
        
	int begin = size * lpuId;
        Range range;
	Range positiveRange = d.getPositiveRange();
        range.min = positiveRange.min + begin;
        range.max = positiveRange.min + begin + size - 1;
        if (lpuId == lpuCount - 1) range.max = positiveRange.max;
	if (lpuId > 0 && frontPadding > 0) {
		range.min = range.min - frontPadding;
		if (range.min < positiveRange.min) {
			range.min = positiveRange.min;
		}
	}
	if (lpuId < lpuCount - 1 && backPadding > 0) {
		range.max = range.max + backPadding;
		if (range.max > positiveRange.max) {
			range.max = positiveRange.max;
		}
	}
	Dimension dimension;
	dimension.range = d.adjustPositiveSubRange(range);
	dimension.setLength();
        return dimension;
}

Dimension block_count_getRange(Dimension d, 
		int lpuCount, 
		int lpuId, 
		bool copyMode, 
		int count, 
		int frontPadding, 
		int backPadding) {
	
	int size = d.length / count;
        int begin = size * lpuId;
        Range range;
	Range positiveRange = d.getPositiveRange();
        range.min = positiveRange.min + begin;
        range.max = positiveRange.min + begin + size - 1;
        if (lpuId == lpuCount - 1) range.max = positiveRange.max;

	if (lpuId > 0 && frontPadding > 0) {
		range.min = range.min - frontPadding;
		if (range.min < positiveRange.min) {
			range.min = positiveRange.min;
		}
	}
	if (lpuId < lpuCount - 1 && backPadding > 0) {
		range.max = range.max + backPadding;
		if (range.max > positiveRange.max) {
			range.max = positiveRange.max;
		}
	}
	Dimension dimension;
	dimension.range = d.adjustPositiveSubRange(range);
	dimension.setLength();
        return dimension;
}

Dimension stride_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode) {

	int perStrideEntries = d.getLength() / lpuCount;
	int myEntries = perStrideEntries;
	int remainder = d.getLength() % lpuCount;
	int extra = 0;
	if (remainder > 0) extra = remainder;
	if (remainder > lpuId) {
		myEntries++;
		extra = lpuId;
	}
	Range range;
	range.min = perStrideEntries * lpuId + extra;
	range.max = range.min + myEntries - 1;
	Dimension dimension;
	dimension.range = d.adjustPositiveSubRange(range);
	dimension.setLength();

	if (!copyMode) {
		return dimension.getNormalizedDimension();
	}

	return dimension;
}
              
Dimension block_stride_getRange(Dimension d, int lpuCount, int lpuId, bool copyMode, int size) {
	
	int stride = size * lpuCount;
	int strideCount = d.getLength() / stride;
	int partialStrideElements = d.getLength() % stride;
	int blockCount = partialStrideElements / size;
	int extraEntriesBefore = partialStrideElements;
	int myEntries = strideCount * size;

	// if extra entries fill up a complete new block in the stride of the current LPU then
	// its number of entries should increase by the size parameter and extra preceeding
	// entries should equal to size * preceeding stride count
	if (blockCount > lpuId) {
		myEntries += size;
		extraEntriesBefore = lpuId * size;
	// If the extra entries does not fill a complete block for the current one then it should
	// have whatever remains after filling up preceeding blocks
	} else if (blockCount == lpuId) {
		myEntries += extraEntriesBefore - blockCount * size;
		extraEntriesBefore = blockCount * size;
	}
	Range range;
	range.min = lpuId * strideCount * stride + extraEntriesBefore;
	range.max = range.min + myEntries - 1;

	Dimension dimension;
	dimension.range = d.adjustPositiveSubRange(range);
	dimension.setLength();

	if (!copyMode) {
		return dimension.getNormalizedDimension();
	}

	return dimension;
}

/********************************* getLPUIdRange functions ***************************************/

LpuIdRange *block_size_getUpperRange(int index, Dimension d, int ppuCount, int size) {
	int cutoff = index / size;
	int lpuCount = d.getLength() / size;
	if (cutoff >= lpuCount - 1) return NULL;
	LpuIdRange *range = new LpuIdRange();
	range->startId = cutoff + 1;
	range->endId = lpuCount - 1;
	return range;
}

LpuIdRange *block_size_getLowerRange(int index, Dimension d, int ppuCount, int size) {
	int cutoff = index / size;
	if (cutoff == 0) return NULL; 	
	LpuIdRange *range = new LpuIdRange();
	range->startId = 0;
	range->endId = cutoff - 1;
	return range;
}

int block_size_getInclusiveLpuId(int index, Dimension d, int ppuCount, int size) {
	return index / size;
}

LpuIdRange *block_count_getUpperRange(int index, Dimension d, int ppuCount, int count) {
	int size = d.getLength() / count;
	int cutoff = index / size;
	if (cutoff >= count - 1) return NULL;
	LpuIdRange *range = new LpuIdRange();
	range->startId = cutoff + 1;
	range->endId = count - 1;
	return range;
}

LpuIdRange *block_count_getLowerRange(int index, Dimension d, int ppuCount, int count) {
	int size = d.getLength() / count;
	int cutoff = index / size;
	if (cutoff == 0) return NULL;
	LpuIdRange *range = new LpuIdRange();
	range->startId = 0;
	range->endId = cutoff - 1;
	return range;
}

int block_count_getInclusiveLpuId(int index, Dimension d, int ppuCount, int count) {
	int size = d.getLength() / count;
	return index / size;
}

LpuIdRange *stride_getUpperRange(int index, Dimension d, int ppuCount) { return NULL; }
LpuIdRange *stride_getLowerRange(int index, Dimension d, int ppuCount) { return NULL; }

int stride_getInclusiveLpuId(int index, Dimension d, int ppuCount) {
	return index % ppuCount;
}
inline LpuIdRange *block_stride_getUpperRange(int index,
                Dimension d, int ppuCount, int size) { return NULL; }
inline LpuIdRange *block_stride_getLowerRange(int index,
                Dimension d, int ppuCount, int size) { return NULL; }

int block_stride_getInclusiveLpuId(int index, Dimension d, int ppuCount, int size) {
	int stride = size * ppuCount;
	int intraStrideIndex = index % stride;
	return intraStrideIndex / size;
}
