/* This test case verifies if we can generate accurate interval descriptions for hierarchical partitions
 * involving one or more index reordering. It embodies several scenarios of partitioning with reordering
 * and generates figures for the one been activated for visual validation.
 **/

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "../utils/list.h"
#include "../structure.h"
#include "../utils/interval.h"
#include "../utils/partition.h"

int mainHIDT() {

	Dimension dim;
	dim.length = 101;
	dim.range.min = 0;
	dim.range.max = 100;

	// scenario #1
	//	BlockSizeInstr blockSize = BlockSizeInstr(dim, 1, 50);
	//	BlockStrideInstr blockStride = BlockStrideInstr(blockSize.getDimension(), 0, 3, 8);
	//	blockStride.setPrevInstr(&blockSize);
	//	StrideInstr stride = StrideInstr(blockStride.getDimension(), 1, 4);
	//	stride.setPrevInstr(&blockStride);
	//	stride.drawIntervals();
	//	stride.drawTrueIntervalDesc(dim, 10);

//		// scenario #2
//		BlockStrideInstr blockStride = BlockStrideInstr(dim, 2, 5, 5);
//		BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 0, 12);
//		blockSize.setPrevInstr(&blockStride);
//		blockSize.drawIntervals();
//		blockSize.drawTrueIntervalDesc(dim, 10);

	//	// scenario #3
	//	BlockStrideInstr blockStride = BlockStrideInstr(dim, 2, 5, 5);
	//	BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 1, 12);
	//	blockSize.setPrevInstr(&blockStride);
	//	blockSize.drawIntervals();
	//	blockSize.drawTrueIntervalDesc(dim, 10);

	//	// scenario #4
	//	StrideInstr stride = StrideInstr(dim, 2, 6);
	//	BlockStrideInstr blockStride = BlockStrideInstr(stride.getDimension(), 0, 2, 4);
	//	blockStride.setPrevInstr(&stride);
	//	blockStride.drawIntervals();
	//	blockStride.drawTrueIntervalDesc(dim, 10);

	//	// scenario #5
	//	BlockCountInstr blockCount = BlockCountInstr(dim, 1,  3);
	//	BlockStrideInstr blockStride = BlockStrideInstr(blockCount.getDimension(), 0, 2, 5);
	//	blockStride.setPrevInstr(&blockCount);
	//	BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 1, 8);
	//	blockSize.setPrevInstr(&blockStride);
	//	StrideInstr stride = StrideInstr(blockSize.getDimension(), 0, 2);
	//	stride.setPrevInstr(&blockSize);
	//	BlockSizeInstr blockSize2 = BlockSizeInstr(stride.getDimension(), 1, 2);
	//	blockSize2.setPrevInstr(&stride);
	//	blockSize2.drawIntervals();
	//	blockCount.drawTrueIntervalDesc(dim, 10);
	//	blockStride.drawTrueIntervalDesc(dim, 10);
	//	blockSize.drawTrueIntervalDesc(dim, 10);
	//	stride.drawTrueIntervalDesc(dim, 10);
	//	blockSize2.drawTrueIntervalDesc(dim, 10);

	//	// scenario #6
	//	BlockSizeInstr blockSize = BlockSizeInstr(dim, 1, 50);
	//	StrideInstr stride = StrideInstr(blockSize.getDimension(), 2, 6);
	//	stride.setPrevInstr(&blockSize);
	//	stride.drawTrueIntervalDesc(dim, 10);

	//	// scenario #7
	//	StrideInstr stride = StrideInstr(dim, 1, 5);
	//	BlockSizeInstr blockSize = BlockSizeInstr(stride.getDimension(), 1, 10);
	//	blockSize.setPrevInstr(&stride);
	//	blockSize.drawTrueIntervalDesc(dim, 10);

	// scenario #8
	BlockStrideInstr blockStride = BlockStrideInstr(dim, 1, 2, 5);
	BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 0, 21);
	blockSize.setPrevInstr(&blockStride);
	blockSize.drawIntervals();
	blockSize.drawTrueIntervalDesc(dim, 10);

	return 0;
}



