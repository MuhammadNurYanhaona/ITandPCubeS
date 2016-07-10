#include "gpu_structure.h"

#include <cuda.h>
#include <cuda_runtime.h>

//------------------------------------------------------ GPU Part Dimension ----------------------------------------------------

__device__ bool GpuDimPartConfig::isIncluded(int index) {
	if (range.min > range.max) {
                return index >= range.max && index <= range.min;
        } else {
                return index >= range.min && index <= range.max;
        }
}

__device__ int GpuDimPartConfig::adjustIndex(int index) {
	if (range.min > range.max)
                return range.min - index;
        else return index + range.min;
}
        
__device__ int GpuDimPartConfig::normalizeIndex(int index) {
	return index - range.min;
}
        
__device__ int GpuDimPartConfig::safeNormalizeIndex(int index, bool matchToMin) {
	int normalIndex = index - range.min;
        int length = (range.max - range.min + 1);
        if (normalIndex >= 0 && normalIndex < length) return normalIndex;
        return (matchToMin) ? 0 : length - 1;
}
