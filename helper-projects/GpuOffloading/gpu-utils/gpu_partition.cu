#include "gpu_partition.h"

#include <cuda.h>
#include <cuda_runtime.h>

//--------------------------------------------------------- Block Size -----------------------------------------------------------/

__device__ int block_size_part_count(int *dimRange, int size) {
	int length = dimRange[1] - dimRange[0] + 1;
	int count = length / size;
	return (count < 1) ? 1 : count; 
}

__device__ void block_size_part_range(int *resultDimRange, int *sourceDimRange,
                int lpuCount, int lpuId,
                int size,
                int frontPadding, int backPadding) {

	int begin = size * lpuId;
	int min = sourceDimRange[0] + begin;
	int max = min + size - 1;
	if (lpuId == lpuCount - 1) max = sourceDimRange[1];
	
	if (lpuId > 0 && frontPadding > 0) {
		min = min - frontPadding;
		if (min < sourceDimRange[0]) min = sourceDimRange[0];
	}
        if (lpuId < lpuCount - 1 && backPadding > 0) {
		max = max + backPadding;
		if (max > sourceDimRange[1]) max = sourceDimRange[1];
	}

	resultDimRange[0] = min;
	resultDimRange[1] = max; 
}

//-------------------------------------------------------- Block Count -----------------------------------------------------------/



//----------------------------------------------------------- Stride -------------------------------------------------------------/



//-------------------------------------------------------- Block Stride ----------------------------------------------------------/
