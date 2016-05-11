#ifndef _H_gpu_utils
#define _H_gpu_utils

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

// Given a set of index ranges to be iterated by an warp that wants to distribute the work among its thread, 
// this routines provides the loop iteration start indexes and step sizes for the threads. Distribution of
// iterations to threads of a warp has a great impact on the performance of a CUDA kernel as a poor choice 
// may lead to loss of parallelism, and increasing bank conflicts and non-coalesced global memory accessess.
// An ideal distribution of iteration indexes should be context sensitive to account for the aforementioned
// issues. For that starter, we adopt the strategy of distributing the lowest range that can support enough
// parallel work for all the threads of a warp.  	 
__device__ void determineLoopIndexesAndSteps(int nestingLevels, 
		int threadId, 
		int *indexRanges, 
		int *indexStartsAndSteps);

// prints the error message, if exists, from the return status of any CUDA operation and exits the program
// if there is an error  
void check_error(cudaError e, std::ofstream &logFile);


#endif
