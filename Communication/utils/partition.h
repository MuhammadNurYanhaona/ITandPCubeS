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
public:
	PartitionInstr(const char *n, Dimension pd, int id, int count, bool r);
	virtual ~PartitionInstr() {}
	void setPrevInstr(PartitionInstr *prevInstr) { this->prevInstr = prevInstr; }
	void setPartId(int partId) { this->partId = partId; }
	bool doesReorderIndex() { return reorderIndex; }
	void drawIntervals();
	List<IntervalSeq*> *getTrueIntervalDesc();
	void drawTrueIntervalDesc(Dimension dimension, int labelGap);
	virtual Dimension getDimension() = 0;
	virtual List<IntervalSeq*> *getIntervalDesc() = 0;
	virtual void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
	virtual int calculatePartsCount(Dimension dimension, bool updateProperties) = 0;
	virtual List<IntervalSeq*> *getIntervalDescForRange(Range idRange) = 0;
	virtual void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) = 0;
};

class BlockSizeInstr : public PartitionInstr {
protected:
	int size;
public:
	BlockSizeInstr(Dimension pd, int id, int size);
	Dimension getDimension();
	List<IntervalSeq*> *getIntervalDesc();
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct);
};

class BlockCountInstr : public PartitionInstr {
protected:
	int count;
public:
	BlockCountInstr(Dimension pd, int id, int count);
	Dimension getDimension();
	List<IntervalSeq*> *getIntervalDesc();
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct);
};

class StrideInstr : public PartitionInstr {
private:
	int ppuCount;
public:
	StrideInstr(Dimension pd, int id, int ppuCount);
	Dimension getDimension();
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
	Dimension getDimension();
	List<IntervalSeq*> *getIntervalDesc();
	void getIntervalDesc(List<IntervalSeq*> *descInConstruct);
	int calculatePartsCount(Dimension dimension, bool updateProperties);
	List<IntervalSeq*> *getIntervalDescForRange(Range idRange);
	void getIntervalDescForRangeHierarchy(List<Range> *idRange, List<IntervalSeq*> *descInConstruct);
};

#endif
