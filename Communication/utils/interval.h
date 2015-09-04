#ifndef INTERVAL_H_
#define INTERVAL_H_

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "list.h"
#include "../structure.h"

static const char INCLUSION_CHAR = '|';
static const char EXCLUSION_CHAR = '_';

class DrawingLine {
private:
	Dimension dim;
	int labelGap;
	List<char> *line;
public:
	DrawingLine(Dimension dim, int labelGap);
	void reset();
	void setIndexToOne(int index);
	void draw();
};

class IntervalSeq {
public:
	int begin;
	int length;
	int period;
	int count;
public:
	IntervalSeq(int b, int l, int p, int c);
	void increaseCount(int amount) {
		count += amount;
	}
	void draw(DrawingLine *drawingLine);
	List<IntervalSeq*> *transformSubInterval(IntervalSeq *subInterval);
	List<IntervalSeq*> *computeIntersection(IntervalSeq *other);
	bool isEqual(IntervalSeq *other);
};

class MultidimensionalIntervalSeq {
protected:
	int dimensionality;
	std::vector<IntervalSeq*> intervals;
public:
	MultidimensionalIntervalSeq(int dimensionality);
	// function to initialize intervals of the current sequence all at once from a template interval vector
	void copyIntervals(std::vector<IntervalSeq*> *templateVector);
	void setIntervalForDim(int dimensionNo, IntervalSeq *intervalSeq);
	IntervalSeq *getIntervalForDim(int dimensionNo);
	List<MultidimensionalIntervalSeq*> *computeIntersection(MultidimensionalIntervalSeq *other);
	bool isEqual(MultidimensionalIntervalSeq *other);
	void draw();
	// function to generate a list of multidimensional interval sequences as a cross-product of lists of
	// one-dimensional interval sequences
	static List<MultidimensionalIntervalSeq*> *generateIntervalSeqs(int dimensionality,
			List<List<IntervalSeq*>*> *intervalSeqLists);
private:
	// a recursive helper routine to generate multidimensional interval sequences
	static List<MultidimensionalIntervalSeq*> *generateIntervalsFromList(int dimensionality,
			List<List<IntervalSeq*>*> *intervalSeqLists,
			std::vector<IntervalSeq*> *constructionInProgress);
};

#endif /* INTERVAL_H_ */
