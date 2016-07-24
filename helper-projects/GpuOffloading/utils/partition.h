#ifndef _H_partition
#define _H_partition

#include "../runtime/structure.h"

#include <iostream>
#include <algorithm>


int block_size_partitionCount(Dimension d, int size);

Dimension block_size_getRange(Dimension d, 
		int lpuCount, int lpuId, 
		int size,
                int frontPadding, int backPadding);

int block_count_partitionCount(Dimension d, int count);

Dimension block_count_getRange(Dimension d, 
		int lpuCount, int lpuId, 
                int frontPadding, int backPadding);
#endif
