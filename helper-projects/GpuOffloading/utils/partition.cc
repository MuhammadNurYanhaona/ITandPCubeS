#ifndef _H_partition
#define _H_partition

#include "../runtime/structure.h"

#include <iostream>
#include <algorithm>


int block_size_partitionCount(Dimension d, int size) {
        return std::max(1, d.length / size);
}

Dimension block_size_getRange(Dimension d,
                int lpuCount,
                int lpuId,
                int size,
                int frontPadding,
                int backPadding) {

        int begin = size * lpuId;
        Range range;
        Range positiveRange = d.getPositiveRange();
        range.min = positiveRange.min + begin;
        range.max = positiveRange.min + begin + size - 1;
        if (lpuId == lpuCount - 1) range.max = positiveRange.max;
        if (lpuId > 0 && frontPadding > 0) {
                range.min = range.min - frontPadding;
                if (range.min < positiveRange.min) {
                        range.min = positiveRange.min;
                }
        }
        if (lpuId < lpuCount - 1 && backPadding > 0) {
                range.max = range.max + backPadding;
                if (range.max > positiveRange.max) {
                        range.max = positiveRange.max;
                }
        }

        Dimension dimension;
        dimension.range = d.adjustPositiveSubRange(range);
        dimension.setLength();
        return dimension;
}

#endif
