/* The goal of this test case is to examine if the partitioning library can generate interval description
 * properly for id-ranges and their hierarchies.
 * */

#include "../structure.h"
#include "../utils/list.h"
#include "../utils/partition.h"
#include "../utils/interval.h"
#include <iostream>
#include <cstdlib>

using namespace std;

int mainIRT() {

	//------------------------------------------------------------ Non hierarchical partition range tests

//	Dimension dim = Dimension();
//	dim.range.min = 0;
//	dim.range.max = 100;
//	dim.setLength();
//	DrawingLine drawingLine = DrawingLine(dim, 10);
//	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;

//	// scenario #1
//	StrideInstr stride = StrideInstr(dim, 0, 6);
//	stride.calculatePartsCount(dim, true);
//	Range range = Range();
//	range.min = 1;
//	range.max = 3;
//	intervalList = stride.getIntervalDescForRange(range);

//	// scenario #2
//	BlockStrideInstr blockStride = BlockStrideInstr(dim, 0, 5, 10);
//	blockStride.calculatePartsCount(dim, true);
//	Range range = Range();
//	range.min = 1;
//	range.max = 3;
//	intervalList = blockStride.getIntervalDescForRange(range);

//	// scenario #3
//	BlockSizeInstr blockSize = BlockSizeInstr(dim, 0, 6);
//	int count = blockSize.calculatePartsCount(dim, true);
//	Range range = Range();
//	range.min = 2;
//	range.max = count - 2;
//	std::cout << "parts count: " << count;
//	intervalList = blockSize.getIntervalDescForRange(range);

	//---------------------------------------------------------------- hierarchical partition range tests

	Dimension bigDim = Dimension();
	bigDim.range.min = 0;
	bigDim.range.max = 999;
	bigDim.setLength();
	DrawingLine drawingLine = DrawingLine(bigDim, 10);
	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
	List<Range> *rangeList = new List<Range>;

//	// scenario #1
//	BlockSizeInstr blockSize = BlockSizeInstr(bigDim, 0, 64);
//	blockSize.calculatePartsCount(bigDim, true);
//	rangeList->Append(Range(1, 9));
//	Dimension nextDimension = blockSize.getDimension();
//	BlockCountInstr blockCount = BlockCountInstr(nextDimension, 0, 4);
//	blockCount.calculatePartsCount(nextDimension, true);
//	blockCount.setPrevInstr(&blockSize);
//	rangeList->Append(Range(1, 2));
//	nextDimension = blockCount.getDimension();
//	StrideInstr stride = StrideInstr(nextDimension, 0, 16);
//	stride.calculatePartsCount(nextDimension, true);
//	stride.setPrevInstr(&blockCount);
//	rangeList->Append(Range(1, 2));
//	stride.getIntervalDescForRangeHierarchy(rangeList, intervalList);

//	// scenario #2
//	StrideInstr stride = StrideInstr(bigDim, 0, 10);
//	rangeList->Append(Range(2, 6));
//	Dimension nextDimension = stride.getDimension();
//	BlockSizeInstr blockSize = BlockSizeInstr(nextDimension, 0, 16);
//	blockSize.setPrevInstr(&stride);
//	rangeList->Append(Range(0, 0));
//	BlockCountInstr blockCount = BlockCountInstr(blockSize.getDimension(), 0, 4);
//	blockCount.setPrevInstr(&blockSize);
//	rangeList->Append(Range(1, 2));
//	blockCount.getIntervalDescForRangeHierarchy(rangeList, intervalList);

	// scenario #3
	BlockStrideInstr blockStride = BlockStrideInstr(bigDim, 0, 10, 8);
	rangeList->Append(Range(0, 1));
	Dimension nextDimension = blockStride.getDimension();
	BlockSizeInstr blockSize = BlockSizeInstr(nextDimension, 1, 25);
	rangeList->Append(Range(1, 2));
	blockSize.setPrevInstr(&blockStride);
	StrideInstr stride = StrideInstr(blockSize.getDimension(), 0, 5);
	stride.setPrevInstr(&blockSize);
	rangeList->Append(Range(3, 4));
	stride.getIntervalDescForRangeHierarchy(rangeList, intervalList);


	//----------------------------------------------------------------------------- Drawing of the result

	cout << "The number of distinct interval sequences: " << intervalList->NumElements();
	for (int i = 0; i < intervalList->NumElements(); i++) {
		intervalList->Nth(i)->draw(&drawingLine);
	}
	drawingLine.draw();
	return 0;
}


