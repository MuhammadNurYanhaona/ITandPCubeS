#include "../utils/list.h"
#include "../utils/interval.h"
#include "../part-management/part_folding.h"
#include "../part-management/part_tracking.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

int mainMDIIT() {

	MultidimensionalIntervalSeq *seq1 = new MultidimensionalIntervalSeq(2);
	IntervalSeq *interval11 = new IntervalSeq(45, 20, 30, 4);
	seq1->setIntervalForDim(0, interval11);
	IntervalSeq *interval12 = new IntervalSeq(15, 30, 30, 1);
	seq1->setIntervalForDim(1, interval12);
	MultidimensionalIntervalSeq *seq2 = new MultidimensionalIntervalSeq(2);
	IntervalSeq *interval21 = new IntervalSeq(2, 4, 10, 15);
	seq2->setIntervalForDim(0, interval21);
	IntervalSeq *interval22 = new IntervalSeq(2, 4, 10, 5);
	seq2->setIntervalForDim(1, interval22);

	List<MultidimensionalIntervalSeq*> *intersect = seq1->computeIntersection(seq2);

	cout << "Sequence 1:------------------------------\n";
	seq1->draw();
	cout << "Sequence 2:------------------------------\n";
	seq2->draw();
	cout << "Intersect:-------------------------------\n";

	Dimension dimension = Dimension();
	dimension.range.min = 0;
	dimension.range.max = 999;
	dimension.length = 1000;
	DrawingLine line1 = DrawingLine(dimension, 10);
	DrawingLine line2 = DrawingLine(dimension, 10);
	if (intersect != NULL) {
		for (int i = 0; i < intersect->NumElements(); i++) {
			MultidimensionalIntervalSeq *seq = intersect->Nth(i);
			seq->getIntervalForDim(0)->draw(&line1);
			seq->getIntervalForDim(1)->draw(&line2);
		}
		line1.draw();
		line2.draw();
	} else {
		cout << "The two intervals do not intersect\n";
	}

	return 0;
}
