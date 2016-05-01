#include "gpu_utils.h"
#include "gpu_constant.h"

#include <cuda.h>
#include <cuda_runtime.h>

__device__ void determineLoopIndexesAndSteps(int nestingLevels,
                int threadId,
                int *indexRanges,
                int *indexStartsAndSteps) {
	
	// Set all indexes to start from the beginning of corresponding range and make an unit step per 
	// iteration. at the same time, determine the last index ranges that can sustain WARP_SIZE number 
	// of parallel threads.
	int lastSustainedParallelLevel = 0;
	for (int i = 0; i < nestingLevels; i++) {
		int position = 2 * i;
		int rangeStart = indexRanges[position];
		int rangeEnd = indexRanges[position + 1];
		int rangeLength = rangeEnd - rangeStart + 1;
		indexStartsAndSteps[position] = indexRanges[position];
		indexStartsAndSteps[position + 1] = 1;
		if (rangeLength >= WARP_SIZE) {
			lastSustainedParallelLevel = i;
		}
	}

	// Set the selected level's iteration to be started from different indexes for different threads 
	// and make the step size to be equal to the WARP_SIZE
	int position = 2 * lastSustainedParallelLevel;
	indexStartsAndSteps[position] += threadId;
	indexStartsAndSteps[position + 1] = WARP_SIZE;
}

