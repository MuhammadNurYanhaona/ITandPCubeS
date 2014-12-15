#ifndef _H_backend_index_xform
#define _H_backend_index_xform

#include "backend_structure.h"

/* For each partition function three index transformation/validation functions are required
   so that the compiler can decide how to access data if the originial data-structure is 
   been stored in a modified order to ensure sequential access within a single LPU partition.

   The signature of these functions are as follows
   1.   bool $(FUNCTION)_xformationNeeded()
	This returns TRUE or FALSE dependending on the need for index transformation for the underlying
	partition function. If transformation is not needed then the next two functions need not be 
	implemented.
   2. 	int $(FUNCTION)_transformIndex(int originialIndex, int LPU_id, int LPU_count, Dimension d, ...)
        Here '...' represents the list of parameters underlying partition function accepts 
        in their appropriate order.
	This returns the transformed index for the original index
   3. 	int $(FUNCTION)_revertIndex(int transformedIndex, int LPU_id, int LPU_count, Dimension d, ...)
	This returns the original index for the transformed index
   4. 	bool $(FUNCTION)_isIndexIncluded(int originialIndex, int LPU_id, int LPU_count, Dimension d, ...)
	This determines if an original index falls within an LPU's data range
*/

/************************************** implementations for block_size partition function */

inline int block_size_xformationNeeded() { false; }

inline bool block_size_isIndexIncluded(int originalIndex, 
		int lpuId, int lpuCount, Dimension d, int size) {
	return originalIndex / size == lpuId;
}

/************************************* implementations for block_count partition function */

inline int block_count_xformationNeeded() { false; }

bool block_count_isIndexIncluded(int originalIndex, 
		int lpuId, int lpuCount, Dimension d, int count) {
	int size = d.range.length() / lpuCount;
	return (originialIndex >= size * lpuId && originalIndex < size * (lpuId + 1));
}

/************************************ implementations for block_stride partition function */

inline int block_stride_xformationNeeded() { true; }

int block_stride_transformIndex(int originalIndex, 
		int lpuId, int lpuCount, Dimension d, int size) {
	int strideNo = originialIndex / (lpuCount * blockSize);
	int strideIndex = originalIndex % (lpuCount * blockSize) - lpuId * size;
	return strideNo * size + strideIndex;
}

int block_stride_revertIndex(int transformedIndex, 
		int lpuId, int lpuCount, Dimension d, int size) {
	int strideNo = transformedIndex / size;
	int strideIndex = transformedIndex % size;
	return strideNo * (size * lpuCount) + lpuId * size + strideIndex;
}

bool block_stride_isIndexIncluded(int originalIndex, 
		int lpuId, int lpuCount, Dimension d, int size) {
	int strideIndex = originalIndex % (size * lpuCount);
	return (strideIndex / size == lpuId);
}

/****************************************** implementations for stride partition function */

inline int stride_xformationNeeded() { true; }

inline int stride_transformIndex(int originalIndex, int strideId, Dimension d) {
        return originalIndex * strideId;
}

inline int stride_revertIndex(int stridedIndex, int strideId, Dimension d) {
        return stridedIndex / strideId;
}

inline bool stride_isIndexIncluded(int originalIndex, int strideId, Dimension d) {
        return (originalIndex % strideId == 0);
}


#endif
