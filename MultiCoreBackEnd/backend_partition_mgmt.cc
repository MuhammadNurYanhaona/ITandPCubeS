#include "backend_partition_mgmt.h"
#include "backend_structure.h"
#include "list.h"

/******************************** partitionCount functions ****************************************/

int block_size_partitionCount(Dimension d, int ppuCount, int size) {
        return d.length / size;
}

int block_count_partitionCount(Dimension d, int ppuCount, int count) {
        return count;
}

int stride_partitionCount(Dimension d, int ppuCount) {
        return ppuCount;
}

int block_stride_partitionCount(Dimension d, int ppuCount, int size) {
        return ppuCount;
}

/*********************************** getRange functions *******************************************/

Dimension *block_size_getRange(Dimension d, int lpuCount, int lpuId, int size, 
		int frontPadding, int backPadding) {
        int begin = size * lpuId;
        Dimension *dimension = new Dimension();
        dimension->range.min = d.range.min + begin;
        dimension->range.max = d.range.min + begin + size - 1;
        if (lpuId == lpuCount - 1) dimension->range.max = d.range.max;
	if (lpuId > 0 && frontPadding > 0) {
		dimension->range.min = dimension->range.min - frontPadding;
		if (dimension->range.min < d.range.min) {
			dimension->range.min = d.range.min;
		}
	}
	if (lpuId < lpuCount - 1 && backPadding > 0) {
		dimension->range.max = dimension->range.max + backPadding;
		if (dimension->range.max > d.range.max) {
			dimension->range.max = d.range.max;
		}
	}
        return dimension;
}

Dimension *block_count_getRange(Dimension d, int lpuCount, int lpuId, int count, 
		int frontPadding, int backPadding) {
	int size = d.length / count;
        int begin = size * lpuId;
        Dimension *dimension = new Dimension();
        dimension->range.min = d.range.min + begin;
        dimension->range.max = d.range.min + begin + size - 1;
        if (lpuId == lpuCount - 1) dimension->range.max = d.range.max;
	if (lpuId > 0 && frontPadding > 0) {
		dimension->range.min = dimension->range.min - frontPadding;
		if (dimension->range.min < d.range.min) {
			dimension->range.min = d.range.min;
		}
	}
	if (lpuId < lpuCount - 1 && backPadding > 0) {
		dimension->range.max = dimension->range.max + backPadding;
		if (dimension->range.max > d.range.max) {
			dimension->range.max = d.range.max;
		}
	}
        return dimension;
}

Dimension *stride_getRange(Dimension d, int lpuCount, int lpuId) {
	int perStrideEntries = d.length / lpuCount;
	int myEntries = perStrideEntries;
	int remainder = d.length % lpuCount;
	int extra = 0;
	if (remainder > 0) extra = remainder;
	if (remainder > lpuId) {
		myEntries++;
		extra = lpuId;
	}
	Dimension *dimension = new Dimension();
	dimension->range.min = perStrideEntries * lpuId + extra;
	dimension->range.max = dimension->range.min + myEntries - 1;
	return dimension;
}
              
Dimension *block_stride_getRange(Dimension d, int lpuCount, int lpuId, int size) {
	int stride = size * lpuCount;
	int strideCount = d.length / stride;
	int partialStrideElements = d.length % stride;
	int blockCount = partialStrideElements / size;
	int extraEntriesBefore;
	int myEntries = strideCount * size;
	if (blockCount > 0) extraEntriesBefore = partialStrideElements;
	if (blockCount > lpuId) {
		myEntries += extraEntriesBefore - blockCount * size;
		extraEntriesBefore = blockCount * size;
	} else if (blockCount == lpuId) {
		myEntries += size;
		extraEntriesBefore = blockCount * size;
	}
	Dimension *dimension = new Dimension();
	dimension->range.min = lpuId * strideCount * size + extraEntriesBefore;
	dimension->range.max = dimension->range.min + myEntries - 1;
	return dimension;
}

/********************************* getLPUIdRange functions ***************************************/

LpuIdRange *block_size_getUpperRange(int index, Dimension d, int ppuCount, int size) {
	int cutoff = index / size;
	int lpuCount = d.length / size;
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
	int size = d.length / count;
	int cutoff = index / size;
	if (cutoff >= count - 1) return NULL;
	LpuIdRange *range = new LpuIdRange();
	range->startId = cutoff + 1;
	range->endId = count - 1;
	return range;
}

LpuIdRange *block_count_getLowerRange(int index, Dimension d, int ppuCount, int count) {
	int size = d.length / count;
	int cutoff = index / size;
	if (cutoff == 0) return NULL;
	LpuIdRange *range = new LpuIdRange();
	range->startId = 0;
	range->endId = cutoff - 1;
	return range;
}

int block_count_getInclusiveLpuId(int index, Dimension d, int ppuCount, int count) {
	int size = d.length / count;
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
