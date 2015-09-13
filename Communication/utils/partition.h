#ifndef PARTITION_H_
#define PARTITION_H_

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "list.h"
#include "../structure.h"
#include "interval.h"

class PartitionInstr {
protected:
	const char *name;
	Dimension parentDim;
	int partId;
	int partsCount;
	PartitionInstr *prevInstr;
	bool reorderIndex;
	bool hasPadding;
	// an attribute to use at runtime to enable/disable padding when calculating the interval description for a range
	// hierarchy representing a folding of parts in a part-container.
	bool excludePaddingInIntervalCalculation;
public:
	PartitionInstr(const char *n, Dimension pd, int id, int count, bool r);
	virtual ~PartitionInstr() {}
	void setPrevInstr(PartitionInstr *prevInstr) { this->prevInstr = prevInstr; }
	void setPartId(int partId) { this->partId = partId; }
	bool doesReorderIndex() { return reorderIndex; }
	void setExcludePaddingFlag(bool stat) { excludePaddingInIntervalCalculation = stat; }
	PartitionInstr *getPreviousInstr() { return prevInstr; }
	int getPartsCount() { return partsCount; }

	void drawIntervals();
	List<IntervalSeq*> *getTrueIntervalDesc();
	void drawTrueIntervalDesc(Dimension dimension, int labelGap);

	// two functions to determine if an interval range completely consumes all indices of the parent's dimension this
	// partition function subdivides
	bool isFilledDimension(Range idRange);
	bool isFilledDimension(Range idRange, Dimension dimension);

	virtual Dimension getDimension(bool includePadding=true) = 0;
	virtual List<IntervalSeq*> *getIntervalDesc() = 0;
	virtual void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
	virtual int calculatePartsCount(Dimension dimension, bool updateProperties) = 0;
	virtual List<IntervalSeq*> *getIntervalDescForRange(Range idRange) = 0;
	virtual void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) = 0;
};

class BlockSizeInstr : public PartitionInstr {
protected:
	int size;
	int frontPadding;
	int rearPadding;
public:
	BlockSizeInstr(Dimension pd, int id, int size);
	Dimension getDimension(bool includePadding=true);
	List<IntervalSeq*> *getIntervalDesc();
	void setPadding(int frontPadding, int rearPadding);
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	IntervalSeq *getPaddinglessIntervalForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct);
};

class BlockCountInstr : public PartitionInstr {
protected:
	int count;
	int frontPadding;
	int rearPadding;
public:
	BlockCountInstr(Dimension pd, int id, int count);
	Dimension getDimension(bool includePadding=true);
	List<IntervalSeq*> *getIntervalDesc();
	void setPadding(int frontPadding, int rearPadding);
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	IntervalSeq *getPaddinglessIntervalForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct);
};

class StrideInstr : public PartitionInstr {
private:
	int ppuCount;
public:
	StrideInstr(Dimension pd, int id, int ppuCount);
	Dimension getDimension(bool includePadding=true);
	List<IntervalSeq*> *getIntervalDesc();
	void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct);
};

class BlockStrideInstr : public PartitionInstr {
private:
	int size;
	int ppuCount;
public:
	BlockStrideInstr(Dimension pd, int id, int ppuCount, int size);
	Dimension getDimension(bool includePadding=true);
	List<IntervalSeq*> *getIntervalDesc();
	void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *idRange, List<IntervalSeq*> *descInConstruct);
};

#endif
