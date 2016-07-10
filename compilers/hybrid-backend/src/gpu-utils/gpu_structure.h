#ifndef _H_gpu_structure
#define _H_gpu_structure

// this header file defines some utility structures needed within CUDA kernel. Some of the structure
// defined here may be analogous to some host level structures. Nevertheless, we have defined them
// here again as needed basis to keep CUDA and C++ utility structures separate. This strategy will 
// allow us to augment any of the analogous pair with functionalities appropriate for the underlying
// target platform without worrying about the consequence in the other platform.

// a structure definition to hold dimension range information of GPU arrays
class GpuRange {
  public:
	int min;
	int max;
};

// a structure definition representing a dimension of a GPU array 
class GpuDimension {
  public:
	GpuRange range;
  public:
	__device__ int getLength() { return range.max - range.min + 1; }	
}; 

// This class holds hierarchical partition information for an array dimension. This structure is mainly
// needed for index transformation of arrays that have been partitioned using some index reordering
// partition function.
class GpuDimPartConfig {
  public:
	int count;
	int index;
	GpuRange range;
	GpuDimPartConfig *parent;
  public:
	__device__ bool isIncluded(int index);
	__device__ int adjustIndex(int index);
	__device__ int normalizeIndex(int index);
	__device__ int safeNormalizeIndex(int index, bool matchToMin);
	
};

#endif
