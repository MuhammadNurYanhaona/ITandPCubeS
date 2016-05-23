#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "list.h"
#include "structures.h"

using namespace std;

static const char INCLUSION_CHAR = '|';
static const char EXCLUSION_CHAR = '_';

class DrawingLine {
private:
	Dimension dim;
	int labelGap;
	List<char> *line;
public:
	DrawingLine(Dimension dim, int labelGap) {
		this->dim = dim; this->labelGap = labelGap;
		line = new List<char>;
		reset();
	}
	void reset() {
		while (line->NumElements() > 0) line->RemoveAt(0);
		for (int i = 0; i < dim.length; i++) line->Append(EXCLUSION_CHAR);
	}
	void setIndexToOne(int index) {
		int positionOfUpdate = index - dim.range.min;
		line->RemoveAt(positionOfUpdate);
		line->InsertAt(INCLUSION_CHAR, positionOfUpdate);
	}
	void draw() {
		cout << "\n";
		for (int i = 0; i < dim.length; i++) {
			cout << line->Nth(i);
		}

		cout << "\n\n";

		int labelPosition = dim.range.min;
		int outputCount = 0;
		while (labelPosition < dim.range.max) {
			cout << labelPosition;
			ostringstream stream;
			stream << labelPosition;
			int labelLength = stream.str().length();
			outputCount += labelLength;
			int fillerChar = labelGap - labelLength;
			for (int i = 0; i < fillerChar && outputCount < dim.length; i++) {
				cout << '.';
				outputCount++;
			}
			labelPosition += labelGap;
		}
		cout << "\n";
	}
};

class IntervalSeq {
public:
	int begin;
	int length;
	int period;
	int count;
public:
	IntervalSeq(int b, int l, int p, int c) {
		begin = b; length = l; period = p; count = c;
	}
	void increaseCount(int amount) {
		count += amount;
	}
	void draw(DrawingLine *drawingLine);
	List<IntervalSeq*> *transformSubInterval(IntervalSeq *subInterval);
};

void IntervalSeq::draw(DrawingLine *drawingLine) {
	for (int interval = 0; interval < count; interval++) {
		int intervalBegin = begin + period * interval;
		int intervalEnd = intervalBegin + length - 1;
		for (int position = intervalBegin; position <= intervalEnd; position++) {
			drawingLine->setIndexToOne(position);
		}
	}
}

List<IntervalSeq*> *IntervalSeq::transformSubInterval(IntervalSeq *subInterval) {
	List<Range> *uniqueRangeList = new List<Range>;
	List<int> *uniqueIntervalBeginnings = new List<int>;

	int subLineSpanEnd = subInterval->begin
			+ (subInterval->count - 1) * subInterval->period
			+ subInterval->length - 1;
	int subIntervalEndingIndex = this->begin
			+ (subLineSpanEnd / this->length) * this->period
			+ subLineSpanEnd % this->length;
	int piecePeriod = subIntervalEndingIndex;

	for (int i = 0; i < subInterval->count; i++) {
		int localBegin = subInterval->begin + subInterval->period * i;
		int parentBegin = localBegin % this->length;
		bool alreadyObserved = false;
		for (int j = 0; j < uniqueIntervalBeginnings->NumElements(); j++) {
			if (uniqueIntervalBeginnings->Nth(j) == parentBegin) {
				alreadyObserved = true;
				break;
			}
		}
		if (alreadyObserved) {
			int firstOccurance = uniqueRangeList->Nth(0).min;
			int firstHolderBlock = firstOccurance / this->length;
			int currentHolderBlock = localBegin / this->length;
			int blockAdvance = (currentHolderBlock - firstHolderBlock);
			piecePeriod = blockAdvance * this->period;
			break;
		}
		uniqueIntervalBeginnings->Append(parentBegin);

		int rangeMin = localBegin;
		int lengthYetToCover = subInterval->length;
		while (lengthYetToCover > 0) {
			int remainingInCurrentInterval = this->length - parentBegin;
			int subLength = min(remainingInCurrentInterval, lengthYetToCover);
			Range range;
			range.min = rangeMin;
			range.max = rangeMin + subLength - 1;
			uniqueRangeList->Append(range);
			lengthYetToCover -= subLength;
			if (lengthYetToCover > 0) parentBegin = 0;
			rangeMin += subLength;
		}
	}

	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
	for (int i = 0; i < uniqueRangeList->NumElements(); i++) {
		Range range = uniqueRangeList->Nth(i);
		int pieceBegin = this->begin
				+ (range.min / this->length) * this->period
				+ range.min % this->length;
		int pieceCount = ((subIntervalEndingIndex - pieceBegin + 1)
				+ piecePeriod - 1) / piecePeriod;
		int pieceLength = range.max - range.min + 1;
		IntervalSeq *intervalPiece = new IntervalSeq(pieceBegin,
				pieceLength, piecePeriod, pieceCount);
		intervalList->Append(intervalPiece);
	}
	return intervalList;
}

class PartitionInstr {
protected:
	const char *name;
	Dimension parentDim;
	int partId;
	int partsCount;
	PartitionInstr *prevInstr;
	bool reorderIndex;
public:
	PartitionInstr(const char *n, Dimension pd, int id, int count, bool r) {
		name = n; parentDim = pd; partId = id; partsCount = count; reorderIndex = r; prevInstr = NULL;
	}
	virtual ~PartitionInstr() {}
	void setPrevInstr(PartitionInstr *prevInstr) {
		this->prevInstr = prevInstr;
	}
	bool doesReorderIndex() { return reorderIndex; }
	void drawIntervals();
	List<IntervalSeq*> *getTrueIntervalDesc();
	void drawTrueIntervalDesc(Dimension dimension, int labelGap);
	virtual Dimension getDimension() = 0;
	virtual List<IntervalSeq*> *getIntervalDesc() = 0;
	virtual void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
};

List<IntervalSeq*> *PartitionInstr::getTrueIntervalDesc() {
	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
	getIntervalDesc(intervalList);
	return intervalList;
}

void PartitionInstr::drawTrueIntervalDesc(Dimension dimension, int labelGap) {
	DrawingLine *drawingLine = new DrawingLine(dimension, labelGap);
	List<IntervalSeq*> *intervals = getTrueIntervalDesc();
	for (int i = 0; i < intervals->NumElements(); i++) {
		intervals->Nth(i)->draw(drawingLine);
	}
	drawingLine->draw();
	while (intervals->NumElements() > 0) {
		IntervalSeq *interval = intervals->Nth(0);
		intervals->RemoveAt(0);
		delete interval;
	}
	delete intervals;
	delete drawingLine;
}

void PartitionInstr::getIntervalDesc(List<IntervalSeq*> *descInConstruct) {
	if (descInConstruct->NumElements() == 0) {
		List<IntervalSeq*> *intervalDesc = getIntervalDesc();
		descInConstruct->AppendAll(intervalDesc);
		while (intervalDesc->NumElements() > 0) intervalDesc->RemoveAt(0);
		delete intervalDesc;
	}
	if (prevInstr != NULL) {
		prevInstr->getIntervalDesc(descInConstruct);
	}
}

void PartitionInstr::drawIntervals() {
	if (prevInstr != NULL) {
		prevInstr->drawIntervals();
	}
	cout << "\n" << name << "\n";
	DrawingLine *drawingLine = new DrawingLine(parentDim, 10);
	List<IntervalSeq*> *intervalList = getIntervalDesc();
	for (int i = 0; i < intervalList->NumElements(); i++) {
		intervalList->Nth(i)->draw(drawingLine);
	}
	drawingLine->draw();
}

class BlockSizeInstr : public PartitionInstr {
protected:
	int size;
public:
	BlockSizeInstr(Dimension pd, int id, int size) : PartitionInstr("Block-Size", pd, id, 0, false) {
		int dimLength = pd.length;
		this->partsCount = (dimLength + size - 1) / size;
		this->size = size;
	}
	Dimension getDimension() {
		int begin = parentDim.range.min + partId * size;
		int remaining = parentDim.range.max - begin + 1;
		int intervalLength = (remaining >= size) ? size : remaining;
		Dimension partDimension;
		partDimension.range.min = begin;
		partDimension.range.max = begin + intervalLength - 1;
		partDimension.length = intervalLength;
		return partDimension;
	}
	List<IntervalSeq*> *getIntervalDesc() {
		List<IntervalSeq*> *list = new List<IntervalSeq*>;
		Dimension partDim = getDimension();
		int begin = partDim.range.min;
		int length = partDim.length;
		int period = length;
		int count = 1;
		IntervalSeq *interval = new IntervalSeq(begin, length, period, count);
		list->Append(interval);
		return list;
	}
};

class BlockCountInstr : public PartitionInstr {
protected:
	int count;
public:
	BlockCountInstr(Dimension pd, int id, int count) : PartitionInstr("Block-Count", pd, id, 0, false) {
		int length = pd.length;
		this->partsCount = max(1, min(count, length));
		this->count = count;
	}
	Dimension getDimension() {
		int size = parentDim.length / count;
		int begin = partId * size;
		int length = (partId < count - 1) ? size : parentDim.range.max - begin + 1;
		Dimension partDimension;
		partDimension.range.min = begin;
		partDimension.range.max = begin + length - 1;
		partDimension.length = length;
		return partDimension;
	}
	List<IntervalSeq*> *getIntervalDesc() {
		List<IntervalSeq*> *list = new List<IntervalSeq*>;
		Dimension partDim = getDimension();
		int begin = partDim.range.min;
		int length = partDim.length;
		int period = length;
		int count = 1;
		IntervalSeq *interval = new IntervalSeq(begin, length, period, count);
		list->Append(interval);
		return list;
	}
};

class StrideInstr : public PartitionInstr {
private:
	int ppuCount;
public:
	StrideInstr(Dimension pd, int id, int ppuCount) : PartitionInstr("Stride", pd, id, 0, true) {
		int length = pd.length;
		this->partsCount = max(1, min(ppuCount, length));
		this->ppuCount = ppuCount;
	}
	Dimension getDimension() {
		int length = parentDim.length;
		int perStrideEntries = length / partsCount;
		int myEntries = perStrideEntries;
		int remainder = length % partsCount;
		if (remainder > partId) {
			myEntries++;
		}
		Dimension partDimension;
		partDimension.range.min = 0;
		partDimension.range.max = myEntries - 1;
		partDimension.length = myEntries;
		return partDimension;
	}
	List<IntervalSeq*> *getIntervalDesc() {
		List<IntervalSeq*> *list = new List<IntervalSeq*>;
		int length = parentDim.length;
		int strides = length /partsCount;
		int remaining = length % partsCount;
		if (remaining > partId) strides++;
		int begin = parentDim.range.min + partId;
		IntervalSeq *interval = new IntervalSeq(begin, 1, partsCount, strides);
		list->Append(interval);
		return list;
	}
	void getIntervalDesc(List<IntervalSeq*> *descInConstruct) {
		IntervalSeq *myIntervalDesc = getIntervalDesc()->Nth(0);
		if (descInConstruct->NumElements() == 0) {
			descInConstruct->Append(myIntervalDesc);
		} else {
			List<IntervalSeq*> *newList = new List<IntervalSeq*>;
			while (descInConstruct->NumElements() > 0) {
				IntervalSeq *currInterval = descInConstruct->Nth(0);
				newList->AppendAll(myIntervalDesc->transformSubInterval(currInterval));
				descInConstruct->RemoveAt(0);
				delete currInterval;
			}
			descInConstruct->AppendAll(newList);
			while (newList->NumElements() > 0) newList->RemoveAt(0);
			delete newList;
		}
		if (prevInstr != NULL) {
			prevInstr->getIntervalDesc(descInConstruct);
		}
	}
};

class BlockStrideInstr : public PartitionInstr {
private:
	int size;
	int ppuCount;
public:
	BlockStrideInstr(Dimension pd, int id, int ppuCount, int size) : PartitionInstr("Block-Stride", pd, id, 0, true) {
		this->size = size;
		this->ppuCount = ppuCount;
		int length = pd.length;
		int strides = length / size;
		partsCount = max(1, min(strides, ppuCount));
	}
	Dimension getDimension() {
		int strideLength = size * partsCount;
		int strideCount = parentDim.length / strideLength;
		int myEntries = strideCount * size;

		int partialStrideElements = parentDim.length % strideLength;
		int blockCount = partialStrideElements / size;
		int extraEntriesBefore = partialStrideElements;

		if (blockCount > partId) {
			myEntries += size;
			extraEntriesBefore = partId * size;
		} else if (blockCount == partId) {
			myEntries += extraEntriesBefore - partId * size;
			extraEntriesBefore = partId * size;
		}

		Dimension partDimension;
		partDimension.range.min = 0;
		partDimension.range.max = myEntries - 1;
		partDimension.length = myEntries;
		return partDimension;
	}
	List<IntervalSeq*> *getIntervalDesc() {

		List<IntervalSeq*> *list = new List<IntervalSeq*>;
		int strideLength = size * partsCount;
		int strideCount = parentDim.length / strideLength;

		int partialStrideElements = parentDim.length % strideLength;
		int extraBlockCount = partialStrideElements / size;

		if (extraBlockCount > partId) strideCount++;

		int begin = parentDim.range.min + size * partId;
		int length = size;
		int count = strideCount;
		int period = strideLength;
		IntervalSeq *iterativeInterval = new IntervalSeq(begin, length, period, count);
		list->Append(iterativeInterval);

		if (extraBlockCount == partId && partialStrideElements % size != 0) {
			int spill = partialStrideElements % size;
			int spillStarts = parentDim.range.min + strideCount * strideLength + extraBlockCount * size;
			IntervalSeq *spillInterval = new IntervalSeq(spillStarts, spill, spill, 1);
			list->Append(spillInterval);
		}

		return list;
	}
	void getIntervalDesc(List<IntervalSeq*> *descInConstruct) {
		List<IntervalSeq*> *newList = new List<IntervalSeq*>;
		List<IntervalSeq*> *myIntervalDescList = getIntervalDesc();
		if (descInConstruct->NumElements() == 0) {
			descInConstruct->AppendAll(myIntervalDescList);
		} else {
			IntervalSeq *myIntervalDesc = myIntervalDescList->Nth(0);
			if (myIntervalDescList->NumElements() > 1) {
				myIntervalDesc->increaseCount(1);
			}
			while (descInConstruct->NumElements() > 0) {
				IntervalSeq *currInterval = descInConstruct->Nth(0);
				newList->AppendAll(myIntervalDesc->transformSubInterval(currInterval));
				descInConstruct->RemoveAt(0);
				delete currInterval;
			}
			descInConstruct->AppendAll(newList);
			while (newList->NumElements() > 0) newList->RemoveAt(0);
			delete newList;
		}
		if (prevInstr != NULL) {
			prevInstr->getIntervalDesc(descInConstruct);
		}
	}
};

int mainIUT() {

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

//	// scenario #2
//	BlockStrideInstr blockStride = BlockStrideInstr(dim, 2, 5, 5);
//	BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 0, 12);
//	blockSize.setPrevInstr(&blockStride);
//	blockSize.drawIntervals();
//	blockSize.drawTrueIntervalDesc(dim, 10);

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

	// scenario #5
	BlockCountInstr blockCount = BlockCountInstr(dim, 1,  3);
	BlockStrideInstr blockStride = BlockStrideInstr(blockCount.getDimension(), 0, 2, 5);
	blockStride.setPrevInstr(&blockCount);
	BlockSizeInstr blockSize = BlockSizeInstr(blockStride.getDimension(), 1, 8);
	blockSize.setPrevInstr(&blockStride);
	StrideInstr stride = StrideInstr(blockSize.getDimension(), 0, 2);
	stride.setPrevInstr(&blockSize);
	BlockSizeInstr blockSize2 = BlockSizeInstr(stride.getDimension(), 1, 2);
	blockSize2.setPrevInstr(&stride);
	blockSize2.drawIntervals();
	blockCount.drawTrueIntervalDesc(dim, 10);
	blockStride.drawTrueIntervalDesc(dim, 10);
	blockSize.drawTrueIntervalDesc(dim, 10);
	stride.drawTrueIntervalDesc(dim, 10);
	blockSize2.drawTrueIntervalDesc(dim, 10);

	return 0;
}



