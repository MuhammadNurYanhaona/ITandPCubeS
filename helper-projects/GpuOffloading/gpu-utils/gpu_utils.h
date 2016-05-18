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

// This function is used to allocate pointers dynamically in the shared memory of an SM. The current CUDA
// framework does not support more than one dynamically allocated (whose size is fixed during the kernel 
// launch time)	shared memory variable. Hence we need to allocate different pointers for data parts of an 
// LPU as taking disjoint ranges from that single dynamic memory.  
__device__ void allocateInSharedMemory(char *memoryPanel, 
		int *panelIndex, 	// a pointer for the index of the next available memory location
		char *pointer, 		// the pointer to be allocated from the memory panel 
		int dataItemSize, 	// actual data type size of the elements of the pointer
		int dataItemCount); 	// total number of items in the pointer


// Allocation of pointers from the dynamic shared memory pannel should always happen at some multiple of the
// following constant to avoid any alignment problem.  
#define MEMORY_PANNEL_ALIGNMENT_BOUNDARY 8

#endif
