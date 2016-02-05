/* The goal of this test case is to verify if we can selectively include and exclude paddings at different
 * level when generating an interval description for a hierarchical partition range.
 * */

#include "../utils/partition.h"
#include "../utils/interval.h"
#include "../utils/list.h"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int mainPIET() {

	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
	Dimension dimension = Dimension();
	dimension.range = Range(0, 99);
	dimension.setLength();
	DrawingLine drawingLine = DrawingLine(dimension, 10);
	List<Range> *rangeList = new List<Range>;

//	// scenario #1
//	BlockSizeInstr blockSize = BlockSizeInstr(dimension, 1, 20);
//	blockSize.setPadding(4, 4);
//	blockSize.drawTrueIntervalDesc(dimension, 10);
//	BlockCountInstr blockCount = BlockCountInstr(blockSize.getDimension(), 0, 4);
//	blockCount.setPrevInstr(&blockSize);
//	blockCount.setPadding(1, 1);
//	blockCount.drawTrueIntervalDesc(dimension, 10);
//	blockSize.setExcludePaddingFlag(true);
//	blockCount.setExcludePaddingFlag(true);
//	rangeList->Append(Range(1, 3));
//	rangeList->Append(Range(0));
//	blockCount.getIntervalDescForRangeHierarchy(rangeList, intervalList);

	// scenario #2
	BlockSizeInstr blockSize = BlockSizeInstr(dimension, 1, 20);
	blockSize.setPadding(5, 5);
	blockSize.drawTrueIntervalDesc(dimension, 10);
	BlockCountInstr blockCount = BlockCountInstr(blockSize.getDimension(), 0, 6);
	blockCount.setPrevInstr(&blockSize);
	blockCount.setPadding(1, 1);
	blockCount.drawTrueIntervalDesc(dimension, 10);
	blockSize.setExcludePaddingFlag(true);
	blockCount.setExcludePaddingFlag(true);
	rangeList->Append(Range(1, 3));
	rangeList->Append(Range(0, 2));
	blockCount.getIntervalDescForRangeHierarchy(rangeList, intervalList);

	cout << "The number of distinct interval sequences: " << intervalList->NumElements();
	for (int i = 0; i < intervalList->NumElements(); i++) {
		intervalList->Nth(i)->draw(&drawingLine);
	}
	drawingLine.draw();

	return 0;
}

