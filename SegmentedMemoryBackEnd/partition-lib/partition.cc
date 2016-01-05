#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "partition.h"
#include "../utils/list.h"
#include "../utils/interval.h"
#include "../codegen/structure.h"

using namespace std;

//------------------------------------------------ Partition Instruction -------------------------------------------------

PartitionInstr::PartitionInstr(const char *n, Dimension pd, int id, int count, bool r) {
	name = n;
	parentDim = pd;
	partId = id;
	partsCount = count;
	reorderIndex = r;
	prevInstr = NULL;
	hasPadding = false;
	excludePaddingInIntervalCalculation = false;
	priorityOrder = 0;
}

PartitionInstr::PartitionInstr(const char *n, bool r) {
	name = n;
	reorderIndex = r;
	parentDim = Dimension();
	partId = -1;
	partsCount = 0;
	prevInstr = NULL;
	hasPadding = false;
	excludePaddingInIntervalCalculation = false;
	priorityOrder = 0;
}

List<IntervalSeq*> *PartitionInstr::getTrueIntervalDesc() {
	List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
	getIntervalDesc(intervalList);
	return intervalList;
}

void PartitionInstr::drawTrueIntervalDesc(Dimension dimension, int labelGap) {
	DrawingLine *drawingLine = new DrawingLine(dimension, labelGap);
	List<IntervalSeq*> *intervals = getTrueIntervalDesc();
	cout << "number of interval sequences: " << intervals->NumElements() << "\n";
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

bool PartitionInstr::isFilledDimension(Range idRange) {
	return partsCount == (idRange.max - idRange.min + 1);
}

bool PartitionInstr::isFilledDimension(Range idRange, Dimension dimension) {
	int count = calculatePartsCount(dimension, false);
	return count == (idRange.max - idRange.min + 1);
}

//--------------------------------------------------- Void Instruction ---------------------------------------------------

VoidInstr::VoidInstr() : PartitionInstr("Void", false) {
	partsCount = 1;
	partId = 0;
}

int VoidInstr::calculatePartsCount(Dimension dimension, bool updateProperties) {
	if (updateProperties) {
		this->parentDim = dimension;
		partsCount = 1;
	}
	return 1;
}

List<IntervalSeq*> *VoidInstr::getIntervalDesc() {
	List<IntervalSeq*> *list = new List<IntervalSeq*>;
	int begin = parentDim.range.min;
	int length = parentDim.length;
	int period = length;
	int count = 1;
	IntervalSeq *interval = new IntervalSeq(begin, length, period, count);
	list->Append(interval);
	return list;
}

void VoidInstr::getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) {
	if (descInConstruct->NumElements() == 0) {
		List<IntervalSeq*> *myIntervalList = getIntervalDesc();
		descInConstruct->AppendAll(myIntervalList);
		delete myIntervalList;
	}
	rangeList->RemoveAt(rangeList->NumElements() - 1);
	if (prevInstr != NULL) prevInstr->getIntervalDescForRangeHierarchy(rangeList, descInConstruct);
}

XformedIndexInfo *VoidInstr::transformIndex(XformedIndexInfo *indexToXform) {
	indexToXform->partNo = 0;
	return NULL;
}

//------------------------------------------------------ Block Size ------------------------------------------------------

BlockSizeInstr::BlockSizeInstr(int size) : PartitionInstr("Block-Size", false) {
	this->size = size;
	frontPadding = 0;
	rearPadding = 0;
}

BlockSizeInstr::BlockSizeInstr(Dimension pd, int id, int size) : PartitionInstr("Block-Size", pd, id, 0, false) {
	int dimLength = pd.length;
	this->partsCount = (dimLength + size - 1) / size;
	this->size = size;
	frontPadding = 0;
	rearPadding = 0;
}

void BlockSizeInstr::setPadding(int frontPadding, int rearPadding) {
	this->frontPadding = frontPadding;
	this->rearPadding = rearPadding;
	if (frontPadding > 0 || rearPadding > 0) {
		hasPadding = true;
	}
}

Dimension BlockSizeInstr::getDimension(bool includePadding) {
	return getDimension(parentDim, partId, partsCount, includePadding);
}

Dimension BlockSizeInstr::getDimension(Dimension parentDim, int partId, int partsCount, bool includePadding) {
	int begin = parentDim.range.min + partId * size;
	int remaining = parentDim.range.max - begin + 1;
	int intervalLength = (remaining >= size) ? size : remaining;
	Dimension partDimension;
	if (includePadding) {
		partDimension.range.min = max(begin - frontPadding, parentDim.range.min);
	} else {
		partDimension.range.min = begin;
	}
	int end = begin + intervalLength - 1;
	if (includePadding) {
		partDimension.range.max = min(end + rearPadding, parentDim.range.max);
	} else {
		partDimension.range.max = end;
	}
	partDimension.setLength();
	return partDimension;
}

List<IntervalSeq*> *BlockSizeInstr::getIntervalDesc() {
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

int BlockSizeInstr::calculatePartsCount(Dimension dimension, bool updateProperties) {
	int count = (dimension.length + size - 1) / size;
	if (updateProperties) {
		this->parentDim = dimension;
		this->partsCount = count;
	}
	return count;
}

List<IntervalSeq*> *BlockSizeInstr::getIntervalDescForRange(Range idRange) {
	List<IntervalSeq*> *list = new List<IntervalSeq*>;
	int partId = this->partId;
	this->partId = idRange.min;
	Dimension startDim = getDimension();
	this->partId = idRange.max;
	Dimension endDim = getDimension();
	Dimension rangeDim = Dimension();
	rangeDim.range.min = startDim.range.min;
	rangeDim.range.max = endDim.range.max;
	rangeDim.setLength();
	IntervalSeq *interval = new IntervalSeq(rangeDim.range.min, rangeDim.length, rangeDim.length, 1);
	list->Append(interval);
	this->partId = partId;
	return list;
}

IntervalSeq *BlockSizeInstr::getPaddinglessIntervalForRange(Range idRange) {
	int begin = parentDim.range.min + idRange.min * size;
	int supposedLength = (idRange.max - idRange.min + 1) * size;
	int remaining = parentDim.range.max - begin + 1;
	int intervalLength = (remaining >= supposedLength) ? supposedLength : remaining;
	return new IntervalSeq(begin, intervalLength, intervalLength, 1);
}

void BlockSizeInstr::getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) {
	Range idRange = rangeList->Nth(rangeList->NumElements() - 1);
	if (descInConstruct->NumElements() == 0) {
		if (excludePaddingInIntervalCalculation) {
			descInConstruct->Append(getPaddinglessIntervalForRange(idRange));
		} else {
			List<IntervalSeq*> *myIntervalList = getIntervalDescForRange(idRange);
			descInConstruct->AppendAll(myIntervalList);
			delete myIntervalList;
		}
	} else {

		// If this partition instruction involves paddings and directions have been given to exclude its padding during
		// interval description for range calculation then a simple way to eliminate padding is to intersect the padding-
		// less interval sequence at this level with the descriptions received from lower levels. Any unwanted padding
		// region included in the lower level calculation will then be chopped off by the intersection.
		if (hasPadding && excludePaddingInIntervalCalculation) {
			IntervalSeq *myInterval = getPaddinglessIntervalForRange(Range(idRange.min));
			List<IntervalSeq*> *originalList = new List<IntervalSeq*>;
			originalList->AppendAll(descInConstruct);
			descInConstruct->clear();
			for (int i = 0; i < originalList->NumElements(); i++) {
				IntervalSeq *seq = originalList->Nth(i);
				List<IntervalSeq*> *intersect = myInterval->computeIntersection(seq);
				if (intersect != NULL) {
					descInConstruct->AppendAll(intersect);
				}
			}
		}

		// the lower level interval description can directly be used if the id range at this level contains a single entry
		if (idRange.max > idRange.min) {
			List<IntervalSeq*> *updatedList = new List<IntervalSeq*>;
			int iterationCount = idRange.max - idRange.min + 1;
			int partId = this->partId;
			this->partId = idRange.min;
			int period = getDimension(false).length;
			for (int i = 0; i < descInConstruct->NumElements(); i++) {
				IntervalSeq *subInterval = descInConstruct->Nth(i);
				// updating the existing sub-interval to iterate multiple times
				if (subInterval->count == 1) {
					subInterval->count = iterationCount;
					subInterval->period = period;
					updatedList->Append(subInterval);
				// generating new smaller intervals for each iteration of the sub-interval
				} else {
					for (int j = 0; j < subInterval->count; j++) {
						int newBegin = subInterval->begin + subInterval->period * j;
						IntervalSeq *newInterval = new IntervalSeq(newBegin, subInterval->length, period, iterationCount);
						updatedList->Append(newInterval);
					}
				}
			}
			this->partId = partId;
			descInConstruct->clear();
			descInConstruct->AppendAll(updatedList);
			delete updatedList;
		}
	}

	if (descInConstruct->NumElements() > 0) {
		rangeList->RemoveAt(rangeList->NumElements() - 1);
		if (prevInstr != NULL) prevInstr->getIntervalDescForRangeHierarchy(rangeList, descInConstruct);
	}
}

XformedIndexInfo *BlockSizeInstr::transformIndex(XformedIndexInfo *indexToXform) {

	Dimension dimension = indexToXform->partDimension;
	int index = indexToXform->index;
	int count = calculatePartsCount(dimension, false);

	bool includePadding = hasPadding && !excludePaddingInIntervalCalculation;
	int partNo = (index - dimension.range.min) / size;
	Dimension newDimension = getDimension(dimension, partNo, count, includePadding);

	indexToXform->partDimension = newDimension;
	indexToXform->index = index;
	indexToXform->partNo = partNo;

	// Note that the logic is that if the content in the index originially belongs to the current part but also 
	// included as in an adjacent part in its overlapping padding region with the current, only then we should
	// return the neigbor as a padding part. If you fail to understand this then the following calculation will 
	// boggle your mind.
	if (includePadding) {
		Dimension paddinglessDim = getDimension(dimension, partNo, count, false);
		int indexForwardDrift = index - paddinglessDim.range.min;
		int indexBackwardDrift = paddinglessDim.range.max - index;

		// the first condition checks if the index originially belongs to the current part
		// the second condition checks if the index is included in the padding region of a neigbor
		// the last condition excludes the terminal case
		if (indexForwardDrift >= 0 && indexForwardDrift <= frontPadding && partNo > 0) {
			XformedIndexInfo *paddingPart = new XformedIndexInfo(index,
					partNo - 1, getDimension(dimension, partNo - 1, count, true));
			return paddingPart;
		} else if (indexBackwardDrift >= 0 && indexBackwardDrift <= rearPadding && partNo < count - 1) {
			XformedIndexInfo *paddingPart = new XformedIndexInfo(index,
					partNo + 1, getDimension(dimension, partNo + 1, count, true));
			return paddingPart;
		}
	}

	return NULL;
}

//----------------------------------------------------- Block Count ------------------------------------------------------

BlockCountInstr::BlockCountInstr(int count) : PartitionInstr("Block-Count", false) {
	this->count = count;
	frontPadding = 0;
	rearPadding = 0;
}

BlockCountInstr::BlockCountInstr(Dimension pd, int id, int count) : PartitionInstr("Block-Count", pd, id, 0, false) {
	int length = pd.length;
	this->partsCount = max(1, min(count, length));
	this->count = count;
	frontPadding = 0;
	rearPadding = 0;
}

void BlockCountInstr::setPadding(int frontPadding, int rearPadding) {
	this->frontPadding = frontPadding;
	this->rearPadding = rearPadding;
	if (frontPadding > 0 || rearPadding > 0) {
		hasPadding = true;
	}
}

Dimension BlockCountInstr::getDimension(bool includePadding) {
	return getDimension(parentDim, partId, partsCount, includePadding);
}

Dimension BlockCountInstr::getDimension(Dimension parentDim, int partId, int partsCount, bool includePadding) {
	int size = parentDim.length / partsCount;
	int begin = parentDim.range.min + partId * size;
	int length = (partId < partsCount - 1) ? size : parentDim.range.max - begin + 1;
	Dimension partDimension;
	if (includePadding) {
		partDimension.range.min = max(begin - frontPadding, parentDim.range.min);
	} else {
		partDimension.range.min = begin;
	}
	int end = begin + length - 1;
	if (includePadding) {
		partDimension.range.max = min(end + rearPadding, parentDim.range.max);
	} else {
		partDimension.range.max = end;
	}
	partDimension.setLength();
	return partDimension;
}

List<IntervalSeq*> *BlockCountInstr::getIntervalDesc() {
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

int BlockCountInstr::calculatePartsCount(Dimension dimension, bool updateProperties) {
	int count = max(1, min(this->count, dimension.length));
	if (updateProperties) {
		this->parentDim = dimension;
		this->partsCount = count;
	}
	return count;
}

List<IntervalSeq*> *BlockCountInstr::getIntervalDescForRange(Range idRange) {
	List<IntervalSeq*> *list = new List<IntervalSeq*>;
	int partId = this->partId;
	this->partId = idRange.min;
	Dimension startDim = getDimension();
	this->partId = idRange.max;
	Dimension endDim = getDimension();
	Dimension rangeDim = Dimension();
	rangeDim.range.min = startDim.range.min;
	rangeDim.range.max = endDim.range.max;
	rangeDim.setLength();
	IntervalSeq *interval = new IntervalSeq(rangeDim.range.min, rangeDim.length, rangeDim.length, 1);
	list->Append(interval);
	this->partId = partId;
	return list;
}

IntervalSeq *BlockCountInstr::getPaddinglessIntervalForRange(Range idRange) {
	int size = parentDim.length / count;
	int begin = parentDim.range.min + idRange.min * size;
	int length = (idRange.max < count - 1)
			? size * (idRange.max - idRange.min + 1) : parentDim.range.max - begin + 1;
	return new IntervalSeq(begin, length, length, 1);
}

void BlockCountInstr::getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) {
	Range idRange = rangeList->Nth(rangeList->NumElements() - 1);
	if (descInConstruct->NumElements() == 0) {
		if (excludePaddingInIntervalCalculation) {
			descInConstruct->Append(getPaddinglessIntervalForRange(idRange));
		} else {
			List<IntervalSeq*> *myIntervalList = getIntervalDescForRange(idRange);
			descInConstruct->AppendAll(myIntervalList);
			delete myIntervalList;
		}
	} else {

		// If this partition instruction involves paddings and directions have been given to exclude its padding during
		// interval description for range calculation then a simple way to eliminate padding is to intersect the padding-
		// less interval sequence at this level with the descriptions received from lower levels. Any unwanted padding
		// region included in the lower level calculation will then be chopped off by the intersection.
		if (hasPadding && excludePaddingInIntervalCalculation) {
			IntervalSeq *myInterval = getPaddinglessIntervalForRange(Range(idRange.min));
			List<IntervalSeq*> *originalList = new List<IntervalSeq*>;
			originalList->AppendAll(descInConstruct);
			descInConstruct->clear();
			for (int i = 0; i < originalList->NumElements(); i++) {
				IntervalSeq *seq = originalList->Nth(i);
				List<IntervalSeq*> *intersect = myInterval->computeIntersection(seq);
				if (intersect != NULL) {
					descInConstruct->AppendAll(intersect);
				}
			}
		}

		// the lower level interval description can directly be used if the id range at this level contains a single entry
		if (idRange.max > idRange.min) {
			List<IntervalSeq*> *updatedList = new List<IntervalSeq*>;
			int iterationCount = idRange.max - idRange.min + 1;
			int partId = this->partId;
			this->partId = idRange.min;
			int period = getDimension(false).length;
			this->partId = partId;
			for (int i = 0; i < descInConstruct->NumElements(); i++) {
				IntervalSeq *subInterval = descInConstruct->Nth(i);
				// updating the existing sub-interval to iterate multiple times
				if (subInterval->count == 1) {
					subInterval->count = iterationCount;
					subInterval->period = period;
					updatedList->Append(subInterval);
					// generating new smaller intervals for each iteration of the sub-interval
				} else {
					for (int j = 0; j < subInterval->count; j++) {
						int newBegin = subInterval->begin + subInterval->period * j;
						IntervalSeq *newInterval = new IntervalSeq(newBegin, subInterval->length, period, iterationCount);
						updatedList->Append(newInterval);
					}
				}
			}
			descInConstruct->clear();
			descInConstruct->AppendAll(updatedList);
			delete updatedList;
		}
	}

	if (descInConstruct->NumElements() > 0) {
		rangeList->RemoveAt(rangeList->NumElements() - 1);
		if (prevInstr != NULL) prevInstr->getIntervalDescForRangeHierarchy(rangeList, descInConstruct);
	}
}

XformedIndexInfo *BlockCountInstr::transformIndex(XformedIndexInfo *indexToXform) {

	Dimension dimension = indexToXform->partDimension;
	int index = indexToXform->index;
	int count = calculatePartsCount(dimension, false);

	int partSize = dimension.length / count;
	int partNo = (index - dimension.range.min) / partSize;
	bool includePadding = hasPadding && !excludePaddingInIntervalCalculation;
	Dimension newDimension = getDimension(dimension, partNo, count, includePadding);

	indexToXform->partDimension = newDimension;
	indexToXform->index = index;
	indexToXform->partNo = partNo;

	// Note that the logic is that if the content in the index originially belongs to the current part but also 
	// included as in an adjacent part in its overlapping padding region with the current, only then we should
	// return the neigbor as a padding part. If you fail to understand this then the following calculation will 
	// boggle your mind.
	if (includePadding) {
		Dimension paddinglessDim = getDimension(dimension, partNo, count, false);
		int indexForwardDrift = index - paddinglessDim.range.min;
		int indexBackwardDrift = paddinglessDim.range.max - index;

		// the first condition checks if the index originially belongs to the current part
		// the second condition checks if the index is included in the padding region of a neigbor
		// the last condition excludes the terminal case
		if (indexForwardDrift >= 0 && indexForwardDrift <= frontPadding && partNo > 0) {
			XformedIndexInfo *paddingPart = new XformedIndexInfo(index,
					partNo - 1, getDimension(dimension, partNo - 1, count, true));
			return paddingPart;
		} else if (indexBackwardDrift >= 0 && indexBackwardDrift <= rearPadding && partNo < count - 1) {
			XformedIndexInfo *paddingPart = new XformedIndexInfo(index,
					partNo + 1, getDimension(dimension, partNo + 1, count, true));
			return paddingPart;
		}
	}

	return NULL;
}

//-------------------------------------------------------- Stride --------------------------------------------------------

StrideInstr::StrideInstr(int ppuCount) : PartitionInstr("Stride", true) {
	this->ppuCount = ppuCount;
}

StrideInstr::StrideInstr(Dimension pd, int id, int ppuCount) : PartitionInstr("Stride", pd, id, 0, true) {
	int length = pd.length;
	this->partsCount = max(1, min(ppuCount, length));
	this->ppuCount = ppuCount;
}

Dimension StrideInstr::getDimension(bool includePadding) {
	return getDimension(parentDim, partId, partsCount, false);
}

Dimension StrideInstr::getDimension(Dimension parentDimension, int partId, int partsCount, bool includePadding) {
	int length = parentDimension.length;
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

List<IntervalSeq*> *StrideInstr::getIntervalDesc() {
	List<IntervalSeq*> *list = new List<IntervalSeq*>;
	int length = parentDim.length;
	int strides = length / partsCount;
	int remaining = length % partsCount;
	if (remaining > partId) strides++;
	int begin = parentDim.range.min + partId;
	IntervalSeq *interval = new IntervalSeq(begin, 1, partsCount, strides);
	list->Append(interval);
	return list;
}

void StrideInstr::getIntervalDesc(List<IntervalSeq*> *descInConstruct) {
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

int StrideInstr::calculatePartsCount(Dimension dimension, bool updateProperties) {
	int count = max(1, min(ppuCount, dimension.length));
	if (updateProperties) {
		this->parentDim = dimension;
		this->partsCount = count;
	}
	return count;
}

List<IntervalSeq*> *StrideInstr::getIntervalDescForRange(Range idRange) {

	List<IntervalSeq*> *list = new List<IntervalSeq*>;

	if (partsCount == idRange.max - idRange.min + 1) {
		int begin = parentDim.range.min;
		int length = parentDim.length;
		int period = length;
		IntervalSeq *interval = new IntervalSeq(begin, length, period, 1);
		list->Append(interval);
	} else {
		int strides = parentDim.length / partsCount;
		int remaining = parentDim.length % partsCount;
		int spillOverEntries = 0;
		if (remaining > idRange.max) strides++;
		else if (remaining > 0) spillOverEntries = max(0, remaining - idRange.min);

		int begin = parentDim.range.min + idRange.min;
		int length = idRange.max - idRange.min + 1;
		IntervalSeq *mainInterval = new IntervalSeq(begin, length, partsCount, strides);
		list->Append(mainInterval);

		if (spillOverEntries > 0) {
			int spillBegin = begin + partsCount * strides;
			int spillLength = spillOverEntries;
			IntervalSeq *spillInterval = new IntervalSeq(spillBegin, spillLength, spillLength, 1);
			list->Append(spillInterval);
		}
	}

	return list;
}

void StrideInstr::getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) {
	Range idRange = rangeList->Nth(rangeList->NumElements() - 1);
	if (descInConstruct->NumElements() == 0) {
		List<IntervalSeq*> *myIntervalList = getIntervalDescForRange(idRange);
		descInConstruct->AppendAll(myIntervalList);
		delete myIntervalList;
	} else {
		List<IntervalSeq*> *newList = new List<IntervalSeq*>;
		int partId = this->partId;
		for (int i = idRange.min; i <= idRange.max; i++) {
			this->partId = i;
			List<IntervalSeq*> *myIntervalList = getIntervalDesc();
			IntervalSeq *myInterval = myIntervalList->Nth(0);
			delete myIntervalList;
			for (int j = 0; j < descInConstruct->NumElements(); j++) {
				IntervalSeq *subSeq = descInConstruct->Nth(j);
				newList->AppendAll(myInterval->transformSubInterval(subSeq));
			}
			delete myInterval;
			delete myIntervalList;
		}
		this->partId = partId;
		while (descInConstruct->NumElements() > 0) {
			IntervalSeq *currSeq = descInConstruct->Nth(0);
			descInConstruct->RemoveAt(0);
			delete currSeq;
		}
		descInConstruct->AppendAll(newList);
		delete newList;
	}
	rangeList->RemoveAt(rangeList->NumElements() - 1);
	if (prevInstr != NULL) prevInstr->getIntervalDescForRangeHierarchy(rangeList, descInConstruct);
}

XformedIndexInfo *StrideInstr::transformIndex(XformedIndexInfo *indexToXform) {

	Dimension dimension = indexToXform->partDimension;
	int index = indexToXform->index;
	int count = calculatePartsCount(dimension, false);

	int zeroBasedIndex = index - dimension.range.min;
	int xformedIndex = zeroBasedIndex / count;
	int partNo = zeroBasedIndex % count;
	Dimension newDimension = getDimension(dimension, partNo, count, false);

	indexToXform->partDimension = newDimension;
	indexToXform->index = xformedIndex;
	indexToXform->partNo = partNo;

	return NULL;
}

//----------------------------------------------------- Block Stride -----------------------------------------------------

BlockStrideInstr::BlockStrideInstr(int ppuCount, int size) : PartitionInstr("Block-Stride", true) {
	this->size = size;
	this->ppuCount = ppuCount;
}

BlockStrideInstr::BlockStrideInstr(Dimension pd, int id,
		int ppuCount, int size) : PartitionInstr("Block-Stride", pd, id, 0, true) {
	this->size = size;
	this->ppuCount = ppuCount;
	int length = pd.length;
	int strides = length / size;
	partsCount = max(1, min(strides, ppuCount));
}

Dimension BlockStrideInstr::getDimension(bool includePadding) {
	return getDimension(parentDim, partId, partsCount, false);
}

Dimension BlockStrideInstr::getDimension(Dimension parentDim, int partId, int partsCount, bool includePadding) {
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

List<IntervalSeq*> *BlockStrideInstr::getIntervalDesc() {

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

void BlockStrideInstr::getIntervalDesc(List<IntervalSeq*> *descInConstruct) {
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

int BlockStrideInstr::calculatePartsCount(Dimension dimension, bool updateProperties) {
	int strides = dimension.length / size;
	int count = max(1, min(strides, ppuCount));
	if (updateProperties) {
		this->parentDim = dimension;
		this->partsCount = count;
	}
	return count;
}

List<IntervalSeq*> *BlockStrideInstr::getIntervalDescForRange(Range idRange) {

	List<IntervalSeq*> *list = new List<IntervalSeq*>;

	if (partsCount == idRange.max - idRange.min + 1) {
		int begin = parentDim.range.min;
		int length = parentDim.length;
		int period = length;
		IntervalSeq *interval = new IntervalSeq(begin, length, period, 1);
		list->Append(interval);
	} else {
		int strideLength = size * partsCount;
		int strideCount = parentDim.length / strideLength;
		int partialStrideElements = parentDim.length % strideLength;
		int extraBlockCount = partialStrideElements / size;
		int spillOver = 0;
		if (extraBlockCount > idRange.max) strideCount++;
		else spillOver = max(0, partialStrideElements - size * idRange.min);

		int begin = parentDim.range.min + size * idRange.min;
		int length = size * (idRange.max - idRange.min + 1);
		int count = strideCount;
		int period = strideLength;
		IntervalSeq *iterativeInterval = new IntervalSeq(begin, length, period, count);
		list->Append(iterativeInterval);

		if (spillOver > 0) {
			int spillStarts = begin + strideCount * strideLength;
			IntervalSeq *spillInterval = new IntervalSeq(spillStarts, spillOver, spillOver, 1);
			list->Append(spillInterval);
		}
	}

	return list;
}

void BlockStrideInstr::getIntervalDescForRangeHierarchy(List<Range> *rangeList, List<IntervalSeq*> *descInConstruct) {
	Range idRange = rangeList->Nth(rangeList->NumElements() - 1);
	if (descInConstruct->NumElements() == 0) {
		List<IntervalSeq*> *myIntervalList = getIntervalDescForRange(idRange);
		descInConstruct->AppendAll(myIntervalList);
		delete myIntervalList;
	} else {
		List<IntervalSeq*> *newList = new List<IntervalSeq*>;
		int partId = this->partId;
		for (int i = idRange.min; i <= idRange.max; i++) {
			this->partId = i;
			List<IntervalSeq*> *myIntervalList = getIntervalDesc();
			IntervalSeq *myInterval = myIntervalList->Nth(0);
			if (myIntervalList->NumElements() > 1) {
				myInterval->increaseCount(1);
			}
			delete myIntervalList;
			for (int j = 0; j < descInConstruct->NumElements(); j++) {
				IntervalSeq *subSeq = descInConstruct->Nth(j);
				newList->AppendAll(myInterval->transformSubInterval(subSeq));
			}
			delete myInterval;
			delete myIntervalList;
		}
		this->partId = partId;
		while (descInConstruct->NumElements() > 0) {
			IntervalSeq *currSeq = descInConstruct->Nth(0);
			descInConstruct->RemoveAt(0);
			delete currSeq;
		}
		descInConstruct->AppendAll(newList);
		delete newList;
	}
	rangeList->RemoveAt(rangeList->NumElements() - 1);
	if (prevInstr != NULL) prevInstr->getIntervalDescForRangeHierarchy(rangeList, descInConstruct);
}

XformedIndexInfo *BlockStrideInstr::transformIndex(XformedIndexInfo *indexToXform) {

	Dimension dimension = indexToXform->partDimension;
	int index = indexToXform->index;
	int count = calculatePartsCount(dimension, false);

	int strideLength = size * count;
	int zeroBasedIndex = index - dimension.range.min;

	int partNo = (zeroBasedIndex % strideLength) / size;
	int xformedIndex = (zeroBasedIndex / strideLength) * size + zeroBasedIndex % size;
	Dimension newDimension = getDimension(dimension, partNo, count, false);

	indexToXform->partDimension = newDimension;
	indexToXform->index = xformedIndex;
	indexToXform->partNo = partNo;

	return NULL;
}
