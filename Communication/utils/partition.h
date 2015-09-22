#ifndef PARTITION_H_
#define PARTITION_H_

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "list.h"
#include "../structure.h"
#include "interval.h"


/* This is an auxiliary data structure definition to be used during transferring data in between communication buffer
 * and the operating memory data-parts. The data parts are stored as a part-hierarchy (see the part-tracking library)
 * and multiple parts may contribute to a single communication buffer and/or receive updated data points from it. We
 * need a mechanism to transform an index in the communication buffer step-by-step into the actual memory location of
 * the memory address holding (or going to receive) the value at the index. This data structure maintains information
 * to traverse the part hierarchy and along the way the iterative transformation of the original index.
 * */
class XformedIndexInfo {
private:
	int index;
	int partNo;
	Dimension partDimension;
};

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

	// this attribute is used to determine the order of partition instructions for a multidimensional data partition;
	// it is by itself not important for the proper functioning of this class
	int priorityOrder;
public:
	PartitionInstr(const char *n, Dimension pd, int id, int count, bool r);
	virtual ~PartitionInstr() {}
	void setPrevInstr(PartitionInstr *prevInstr) { this->prevInstr = prevInstr; }
	void setPartId(int partId) { this->partId = partId; }
	bool doesReorderIndex() { return reorderIndex; }
	void setExcludePaddingFlag(bool stat) { excludePaddingInIntervalCalculation = stat; }
	PartitionInstr *getPreviousInstr() { return prevInstr; }
	int getPartsCount() { return partsCount; }
	void setPriorityOrder(int priorityOrder) { this->priorityOrder = priorityOrder; }
	int getPriorityOrder() {return priorityOrder; }

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
