#include "index_xform.h"
#include "../runtime/structure.h"

bool block_count_isIndexIncluded(int originalIndex,
                int lpuId, int lpuCount, Dimension d, int count) {
        int size = d.getLength() / lpuCount;
        return (originalIndex >= size * lpuId && originalIndex < size * (lpuId + 1));
}

int block_stride_transformIndex(int originalIndex,
                int lpuId, int lpuCount, Dimension d, int size) {
        int strideNo = originalIndex / (lpuCount * size);
        int strideIndex = originalIndex % (lpuCount * size) - lpuId * size;
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
