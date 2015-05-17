#include "interval_utils.h"
#include "list.h"

/* Implementation file for interval sequence specification management library
   @primary author: Nathan Brunelle
   @supporing functions added by: Andrew Grimshaw & Muhammad Yanhaona
*/

//------------------------------------------------ Line Interval ------------------------------------------------/

LineInterval::LineInterval(int begin, int length, int count, int gap) {
	this->begin = begin;
	this->length = length;
	this->count = count;
	this->gap = gap;
	this->line = NULL;
}

bool LineInterval::isEqual(LineInterval *other) {
	if (this->begin == other->begin 
			&& this->length == other->length 
			&& this->count == other->count 
			&& this->gap == other->gap) {
		return this->getTotalElements() == other->getTotalElements();
	}
	return false;
}

LineInterval *LineInterval::join(LineInterval *other) {
	if (this->begin + length != other->begin) return NULL;
	if (this->gap != other->gap) return NULL;
	if (this->count != other->count) return NULL;
	int newLength = this->length + other->length;
	if (this->count > 0 && this->gap < newLength) return NULL;
	LineInterval *composite = new LineInterval(this->begin, newLength, this->count, this->gap);
	return composite;
}

LineInterval *LineInterval::getFullLineInterval(Line *line) {
	return new LineInterval(line->getStart(), line->getLength(), 1, 0);
}

//-------------------------------------------- Hyperplane Interval ---------------------------------------------/

bool HyperplaneInterval::isEqual(HyperplaneInterval *other) {
	if (this->dimensions != other->dimensions) return false;
	for (int i = 0; i < dimensions; i++) {
		if (!this->lineIntervals->Nth(i)->isEqual(other->lineIntervals->Nth(i))) {
			return false;
		}
	}
	return true;
}

HyperplaneInterval *HyperplaneInterval::join(HyperplaneInterval *other) {
	int unequalDimensionCount = 0;
	int nonmatchingDimension = -1;
	if (this->dimensions != other->dimensions) return NULL;
	for (int i = 0; i < dimensions; i++) {
		if (!this->lineIntervals->Nth(i)->isEqual(other->lineIntervals->Nth(i))) {
			nonmatchingDimension = i;
			unequalDimensionCount++;
		}
	}
	if (unequalDimensionCount != 1) return NULL;
	LineInterval *joinedLineInterval = this->lineIntervals->Nth(nonmatchingDimension)->join(
			other->lineIntervals->Nth(nonmatchingDimension));
	if (joinedLineInterval == NULL) return NULL;

	List<LineInterval*> *intervalList = new List<LineInterval*>;
	intervalList->AppendAll(this->lineIntervals);
	intervalList->RemoveAt(nonmatchingDimension);
	intervalList->InsertAt(joinedLineInterval, nonmatchingDimension);
	return new HyperplaneInterval(this->dimensions, intervalList);
}

//----------------------------------------------- Interval Set -------------------------------------------------/

bool IntervalSet::contains(HyperplaneInterval *interval) {
	for (int i = 0; i < intervals->NumElements(); i++) {
		if (intervals->Nth(i)->isEqual(interval)) return true;
	}
	return false;
}

void IntervalSet::add(HyperplaneInterval *interval) {
	if (!contains(interval)) intervals->Append(interval);
}

void IntervalSet::remove(HyperplaneInterval *interval) {
	if (contains(interval)) {
		int index = 0;
		for (; index < intervals->NumElements(); index++) {
			if (intervals->Nth(index)->isEqual(interval)) break;
		}
		intervals->RemoveAt(index);
	}
}

void IntervalSet::clear() {
	while (intervals->NumElements() > 0) intervals->RemoveAt(0);
}

