#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "list.h"
#include "../structure.h"
#include "interval.h"

using namespace std;

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

