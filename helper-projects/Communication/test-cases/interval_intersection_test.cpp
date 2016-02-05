#include "../utils/list.h"
#include "../utils/interval.h"
#include "../part-management/part_folding.h"
#include "../part-management/part_tracking.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

int mainIIT() {

	Dimension dim;
	dim.length = 8;
	dim.range.min = 0;
	dim.range.max = 7;

	// scenario #1
	//IntervalSeq *interval1 = new IntervalSeq(10, 5, 20, 4);
	//IntervalSeq *interval2 = new IntervalSeq(0, 2, 10, 5);

	// scenario #2
	IntervalSeq *interval1 = new IntervalSeq(0, 8, 8, 1);
	IntervalSeq *interval2 = new IntervalSeq(6, 2, 8, 1);

	// scenario #3
	// IntervalSeq *interval1 = new IntervalSeq(28, 6, 6, 1);
	// IntervalSeq *interval2 = new IntervalSeq(2, 10, 20, 5);

	// scenario #4
	//IntervalSeq *interval1 = new IntervalSeq(5, 5, 15, 10);
	//IntervalSeq *interval2 = new IntervalSeq(38, 5, 6, 10);

	// scenario #5
	//IntervalSeq *interval1 = new IntervalSeq(0, 2, 10, 10);
	//IntervalSeq *interval2 = new IntervalSeq(2, 2, 10, 10);

	// scenario #6
	//IntervalSeq *interval1 = new IntervalSeq(0, 3, 10, 15);
	//IntervalSeq *interval2 = new IntervalSeq(2, 2, 10, 10);

	// scenario #7
	//IntervalSeq *interval1 = new IntervalSeq(45, 20, 30, 4);
	//IntervalSeq *interval2 = new IntervalSeq(2, 4, 10, 15);

	List<IntervalSeq*> *intersection = interval1->computeIntersection(interval2);
	DrawingLine *drawLine = new DrawingLine(dim, 10);
	cout << "First interval Sequence:";
	interval1->draw(drawLine);
	drawLine->draw();
	drawLine->reset();
	cout << "Second Interval Sequence:";
	interval2->draw(drawLine);
	drawLine->draw();
	drawLine->reset();
	cout << "Intersection:";
	if (intersection != NULL) {
		for (int i = 0; i < intersection->NumElements(); i++) {
			intersection->Nth(i)->draw(drawLine);
		}
		drawLine->draw();
	} else {
		cout << "The intervals do not intersect\n";
	}
	return 0;
}


