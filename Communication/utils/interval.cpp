#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <math.h>
#include "list.h"
#include "../structure.h"
#include "interval.h"
#include "common.h"

using namespace std;

//------------------------------------------------------------- Drawing Line ------------------------------------------------------------/

DrawingLine::DrawingLine(Dimension dim, int labelGap) {
	this->dim = dim; this->labelGap = labelGap;
	line = new List<char>;
	reset();
}

void DrawingLine::reset() {
	while (line->NumElements() > 0) line->RemoveAt(0);
	for (int i = 0; i < dim.length; i++) line->Append(EXCLUSION_CHAR);
}

void DrawingLine::setIndexToOne(int index) {
	int positionOfUpdate = index - dim.range.min;
	line->RemoveAt(positionOfUpdate);
	line->InsertAt(INCLUSION_CHAR, positionOfUpdate);
}

void DrawingLine::draw() {
	cout << "\nA bar represents a 1 and a flat line represents a 0\n\n";
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

//--------------------------------------------------------- 1D Interval Sequence --------------------------------------------------------/

IntervalSeq::IntervalSeq(int b, int l, int p, int c) {
	begin = b;
	length = l;
	period = p;
	count = c;
}

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

	// optimization for a common special case
	if (subInterval->count == 1) {
		int fullIterations = subInterval->length / this->length;
		int extraEntries = subInterval->length % this->length;
		List<IntervalSeq*> *intervalList = new List<IntervalSeq*>;
		if (fullIterations > 0) {
			int pieceBegin = this->begin + (subInterval->begin / this->length) * this->period
					+ subInterval->begin % this->length;
			IntervalSeq *fullIterationSequence = new IntervalSeq(pieceBegin,
					this->length, this->period, fullIterations);
			intervalList->Append(fullIterationSequence);
		}
		if (extraEntries > 0) {
			int extraEntryStart = subInterval->begin + fullIterations * this->length;
			int pieceBegin = this->begin + (extraEntryStart / this->length) * this->period
					+ extraEntryStart % this->length;
			IntervalSeq *partialIteration =
					new IntervalSeq(pieceBegin, extraEntries, extraEntries, 1);
			intervalList->Append(partialIteration);
		}
		return intervalList;
	}

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
		int pieceBegin = this->begin + (range.min / this->length) * this->period
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

List<IntervalSeq*> *IntervalSeq::computeIntersection(IntervalSeq *other) {

	List<IntervalSeq*> *intersect = new List<IntervalSeq*>;

	// check for equality fist and return the current interval if the two are the same
	if (this->isEqual(other)) {
		intersect->Append(this);
		return intersect;
	}

	IntervalSeq *first = this;
	IntervalSeq *second = other;
	if (other->count == 1) { first = other; second = this; }

	// declare short-hand variables for interval properties
	int c1 = first->count;
	int c2 = second->count;
	int p1 = first->period;
	int p2 = second->period;
	int l1 = first->length;
	int l2 = second->length;
	int b1 = first->begin;
	int b2 = second->begin;

	// find the ending index for both intervals
	int e1 = b1 + p1 * (c1 - 1) + l1 - 1;
	int e2 = b2 + p2 * (c2 - 1) + l2 - 1;

	// one interval begins after the other ends then there is no intersection
	if (b1 > e2 || b2 > e1) return NULL;

	// skip iterations from one sequence that finishes before the beginning of the other
	int i1 = (b2 >= (b1 + l1)) ? (b2 - b1) / p1 : 0;
	int i2 = (b1 >= (b2 + l2)) ? (b1 - b2) / p2 : 0;
	int bi1 = i1 * p1 + b1;
	int bi2 = i2 * p2 + b2;

	// if the two periods are the same and one is drifted from the other by a sufficient margin then the
	// two interval sequences never intersect
	int drift = abs(bi1 - bi2);
	if ((p1 == p2)
			&& ((bi1 > bi2 && l2 < drift)
					|| (bi2 > bi1 && l1 < drift))) return NULL;

	// take care of the common terminal case where one of the interval sequences iterates just once
	if (c1 == 1) {

		// describe the partial overlapping between the two interval beginnings, if exists
		if (bi2 < bi1 && bi2 + l2 > bi1) {
			int overlapLength = min(bi2 + l2, bi1 + l1) - bi1;
			IntervalSeq *startingOverlap = new IntervalSeq(bi1, overlapLength, overlapLength, 1);
			intersect->Append(startingOverlap);
		}

		// describe iterations of the second sequence that complete within the confinement of the first
		int fullIntervalStart = (bi2 >= bi1) ? i2 : i2 + 1;
		int fullIntervalEnd = min((int) floor((e1 - (b2 + l2 - 1)) * 1.0 / p2), c2 - 1);
		int intervalCount = max(fullIntervalEnd - fullIntervalStart + 1, 0);
		if (intervalCount > 0) {
			int begin = fullIntervalStart * p2 + b2;
			IntervalSeq *fullIterations = new IntervalSeq(begin, l2, p2, intervalCount);
			intersect->Append(fullIterations);
		}

		// describe the partial overlapping between the ending of the first one with some iteration
		// of the second, if exists
		int endIntervalNo = (e1 - b2) / p2;
		int endIntervalBegin = endIntervalNo * p2 + b2;
		if (endIntervalNo < c2 && endIntervalBegin <= e1 && endIntervalBegin + l2 - 1 > e1) {
			int overlapLength = e1 - max(b1, endIntervalBegin) + 1;
			IntervalSeq *endingOverlap = new IntervalSeq(endIntervalBegin,
					overlapLength, overlapLength, 1);
			intersect->Append(endingOverlap);
		}

		if (intersect->NumElements() == 0) {
			delete intersect;
			return NULL;
		}
		return intersect;
	}

	// after LCM number of iterations the drift between the beginnings of the next intervals of the
	// two sequences will be the same; so the intersection calculation algorithm needs to consider
	// only those intervals that may appear within the LCM
	int LCM = lcm(p1, p2);
	int c1L = LCM / p1;
	int c2L = LCM / p2;

	// we need to keep track of the intersecting ranges to later from sequences
	List<Range> *ranges = new List<Range>;

	// initiate counters and interval starting indexes for overlapping range detection process
	int b1i = bi1;
	int ci1 = 0;
	int b2i = bi2;
	int ci2 = 0;

	while (ci1 < c1L || ci2 < c2L) {
		// record any possible overlapping in current iterations
		if (b1i >= b2i) {
			if (b2i + l2 > b1i) {
				ranges->Append(Range(b1i, min(b2i + l2, b1i + l1) - 1));
			}
		} else {
			if (b1i + l1 > b2i) {
				ranges->Append(Range(b2i, min(b2i + l2, b1i + l1) - 1));
			}
		}
		// advance the sequence whose iteration finishes first
		if (b1i + l1 <= b2i + l2) {
			b1i += p1;
			ci1++;
		} else {
			b2i += p2;
			ci2++;
		}
	}

	// generate interval sequences from the range list by determining how many time an overlapping
	// range can appear before the first interval sequence ends
	int earlierEnding = min(e1, e2);
	int period = LCM;
	for (int i = 0; i < ranges->NumElements(); i++) {
		Range range = ranges->Nth(i);
		int begin = range.min;
		int length = range.max - range.min + 1;
		int count = (earlierEnding - range.max) / period + 1;
		if (count > 0) {
			int partPeriod = (count == 1) ? length : period;
			IntervalSeq *interval = new IntervalSeq(begin, length, partPeriod, count);
			intersect->Append(interval);
		}
	}

	if (intersect->NumElements() == 0) {
		delete intersect;
		return NULL;
	}
	return intersect;
}

bool IntervalSeq::isEqual(IntervalSeq *other) {
	return (this->begin == other->begin
			&& this->length == other->length
			&& this->period == other->period
			&& this->count 	== other->count);
}

//-------------------------------------------------- Multidimensional Interval Sequence  ------------------------------------------------/

MultidimensionalIntervalSeq::MultidimensionalIntervalSeq(int dimensionality) {
	this->dimensionality = dimensionality;
	intervals = vector<IntervalSeq*>();
	intervals.reserve(dimensionality);
	for (int i = 0; i < dimensionality; i++) {
		intervals[i] = NULL;
	}
}

void MultidimensionalIntervalSeq::copyIntervals(vector<IntervalSeq*> *templateVector) {
	intervals = vector<IntervalSeq*>(*templateVector);
}

void MultidimensionalIntervalSeq::setIntervalForDim(int dimensionNo, IntervalSeq *intervalSeq) {
	intervals[dimensionNo] = intervalSeq;
}

IntervalSeq *MultidimensionalIntervalSeq::getIntervalForDim(int dimensionNo) {
	return intervals[dimensionNo];
}

bool MultidimensionalIntervalSeq::isEqual(MultidimensionalIntervalSeq *other) {
	for (int i = 0; i < dimensionality; i++) {
		if (!intervals[i]->isEqual(other->intervals[i])) return false;
	}
	return true;
}

List<MultidimensionalIntervalSeq*> *MultidimensionalIntervalSeq::computeIntersection(MultidimensionalIntervalSeq *other) {
	List<List<IntervalSeq*>*> *dimensionalIntersects = new List<List<IntervalSeq*>*>;
	for (int i = 0; i < dimensionality; i++) {
		List<IntervalSeq*> *intersect = intervals[i]->computeIntersection(other->intervals[i]);
		if (intersect == NULL || intersect->NumElements() == 0) {
			while (dimensionalIntersects->NumElements() > 0) {
				List<IntervalSeq*> *subIntersect = dimensionalIntersects->Nth(0);
				delete subIntersect;
			}
			delete dimensionalIntersects;
			return NULL;
		}
		dimensionalIntersects->Append(intersect);
	}
	vector<IntervalSeq*> *constructionVector = new vector<IntervalSeq*>;
	List<MultidimensionalIntervalSeq*> *intersect =
			MultidimensionalIntervalSeq::generateIntervalsFromList(dimensionality,
					dimensionalIntersects, constructionVector);
	delete constructionVector;
	delete dimensionalIntersects;
	return intersect;
}

List<MultidimensionalIntervalSeq*> *MultidimensionalIntervalSeq::generateIntervalSeqs(int dimensionality,
		List<List<IntervalSeq*>*> *intervalSeqLists) {
	vector<IntervalSeq*> *constructionVector = new vector<IntervalSeq*>;
	List<MultidimensionalIntervalSeq*> *mdIntervalSeqList =
			generateIntervalsFromList(dimensionality, intervalSeqLists, constructionVector);
	delete constructionVector;
	return mdIntervalSeqList;
}

List<MultidimensionalIntervalSeq*> *MultidimensionalIntervalSeq::generateIntervalsFromList(int dimensionality,
		List<List<IntervalSeq*>*> *intervalSeqLists,
		std::vector<IntervalSeq*> *constructionInProgress) {

	int position = constructionInProgress->size();
	List<IntervalSeq*> *myList = intervalSeqLists->Nth(position);
	List<MultidimensionalIntervalSeq*> *result = new List<MultidimensionalIntervalSeq*>;
	for (int i = 0; i < myList->NumElements(); i++) {
		IntervalSeq *sequence = myList->Nth(i);
		constructionInProgress->push_back(sequence);
		if (position < dimensionality - 1) {
			List<MultidimensionalIntervalSeq*> *resultPart =
					generateIntervalsFromList(dimensionality, intervalSeqLists, constructionInProgress);
			result->AppendAll(resultPart);
			delete resultPart;
		} else {
			MultidimensionalIntervalSeq *multIntervalSeq = new MultidimensionalIntervalSeq(dimensionality);
			multIntervalSeq->copyIntervals(constructionInProgress);
			result->Append(multIntervalSeq);
		}
		constructionInProgress->erase(constructionInProgress->begin() + position);
	}
	return result;
}

void MultidimensionalIntervalSeq::draw() {
	for (int i = 0; i < dimensionality; i++) {
		IntervalSeq *interval = intervals[i];
		int begin = 0;
		int end = interval->begin + interval->period * interval->count;
		Dimension dim = Dimension();
		dim.range.min = begin;
		dim.range.max = end;
		dim.length = end - begin + 1;
		DrawingLine drawLine = DrawingLine(dim, 10);
		interval->draw(&drawLine);
		cout << "Dimension No: " << i + 1;
		drawLine.draw();
	}
}
