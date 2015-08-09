#ifndef INTERVAL_H_
#define INTERVAL_H_

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "list.h"
#include "structure.h"

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
};

#endif /* INTERVAL_H_ */
