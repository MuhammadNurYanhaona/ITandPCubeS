#include "gpu_utils.h"
#include "gpu_constant.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>

// Notice that in all device function implementations we try to avoid declaring local variables as much 
// as possible as CUDA merge device functions into the calling kernels that will result in too many local
// variables per thread.

__device__ void determineLoopIndexesAndSteps(int nestingLevels,
                int threadId,
                int *indexRanges,
                int *indexStartsAndSteps) {
	
	// Set all indexes to start from the beginning of corresponding range and make an unit step per 
	// iteration. at the same time, determine the last index ranges that can sustain WARP_SIZE number 
	// of parallel threads.
	int lastSustainedParallelLevel = 0;
	for (int i = 0; i < nestingLevels; i++) {
		indexStartsAndSteps[2 * i] = indexRanges[2 * i];
		indexStartsAndSteps[2 * i + 1] = 1;
		if ((indexRanges[2 * i + 1] - indexRanges[2 * i] + 1) >= WARP_SIZE) {
			lastSustainedParallelLevel = i;
		}
	}

	// Set the selected level's iteration to be started from different indexes for different threads 
	// and make the step size to be equal to the WARP_SIZE
	indexStartsAndSteps[2 * lastSustainedParallelLevel] += threadId;
	indexStartsAndSteps[2 * lastSustainedParallelLevel + 1] = WARP_SIZE;
}

void check_error(cudaError e, std::ofstream &logFile) {
        if (e != cudaSuccess) {
                logFile << "CUDA error: " <<  cudaGetErrorString(e) << "\n";
		logFile.flush();
                std::exit(EXIT_FAILURE);
        }
}

__device__ void allocateInSharedMemory(char *memoryPanel, int *panelIndex,
                char *pointer, 
		int dataItemSize, int dataItemCount) {

	// allocate the pointer at the current position of the panel index
	pointer = &memoryPanel[*panelIndex];

	// advance the panel index to the next allocation point; note that we always keep the allocation
	// index at some alignment boundary
	*panelIndex += ((dataItemSize * dataItemCount) / MEMORY_PANNEL_ALIGNMENT_BOUNDARY) 
			* MEMORY_PANNEL_ALIGNMENT_BOUNDARY;
	if (((dataItemSize * dataItemCount) % MEMORY_PANNEL_ALIGNMENT_BOUNDARY) != 0) {
		*panelIndex += MEMORY_PANNEL_ALIGNMENT_BOUNDARY;	
	}
}
