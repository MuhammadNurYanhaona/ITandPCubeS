#ifndef _H_gpu_partition
#define _H_gpu_partition

/* This library has LPU count and partition range determining routines for the partition functions currently supported
 * in IT. Although we have the routines for the same purpose already available for host LPUs, we need to develop equi-
 * valent versions for the GPU LPUs for two reasons. First, the data structures used in the host has member functions
 * used during the count and range determination that are not accessible from the GPU code, requiring the routines have 
 * simpler alternative arguments. Second, the count and range determining functions themselves are not accessible from
 * the GPU as they are not, and cannot be declared, device functions.
 */

// TODO: note that the current implementations of these routines assume that the dimension being partitioned increases
// in the positive direction. This restriction needs to be relaxed in the future as IT also supports dimension ranges
// that advances in the negative direction, i.e., starting at a larger index and ending to a smaller index.

#include <cuda.h>
#include <cuda_runtime.h>

//--------------------------------------------------------------------------------------- Block Size Partition Function

__device__ int block_size_part_count(int *dimRange, int ppuCount, int size);

__device__ void block_size_part_range(int *resultDimRange, int *sourceDimRange, 
		int lpuCount, int lpuId, 
		int size,
                int frontPadding, int backPadding);

//-------------------------------------------------------------------------------------- Block Count Partition Function

#endif
