#include "gpu_partition.h"
#include "gpu_structure.h"

#include <cuda.h>
#include <cuda_runtime.h>

//--------------------------------------------------------- Block Size -----------------------------------------------------------/

__device__ int block_size_part_count(GpuDimension *dimension, int ppuCount, int size) {
	int length = dimension->range.max - dimension->range.min + 1;
	return (length + size - 1) / size;
}

__device__ void block_size_part_range(GpuDimension *resultDimension, 
		GpuDimension *sourceDimension,
                int lpuCount, int lpuId,
                int size,
                int frontPadding, int backPadding) {

	int begin = size * lpuId;
	int min = sourceDimension->range.min + begin;
	int max = min + size - 1;
	if (lpuId == lpuCount - 1) max = sourceDimension->range.max;
	
	if (lpuId > 0 && frontPadding > 0) {
		min = min - frontPadding;
		if (min < sourceDimension->range.min) min = sourceDimension->range.min;
	}
        if (lpuId < lpuCount - 1 && backPadding > 0) {
		max = max + backPadding;
		if (max > sourceDimension->range.max) max = sourceDimension->range.max;
	}

	resultDimension->range.min = min;
	resultDimension->range.max = max; 
}

__device__ void block_size_part_range(GpuDimension *resultDimension, 
		GpuDimension *sourceDimension,
                int lpuCount, int lpuId, int size) {
	block_size_part_range(resultDimension, 
			sourceDimension, lpuCount, lpuId, size, 0, 0);
}

//-------------------------------------------------------- Block Count -----------------------------------------------------------/



//----------------------------------------------------------- Stride -------------------------------------------------------------/



//-------------------------------------------------------- Block Stride ----------------------------------------------------------/
