#include "../utils/list.h"
#include "../utils/interval.h"

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int mainIDCT() {

	List<MultidimensionalIntervalSeq*> *list1 = new List<MultidimensionalIntervalSeq*>;

	MultidimensionalIntervalSeq *seq1 = new MultidimensionalIntervalSeq(2);
	seq1->setIntervalForDim(0, new IntervalSeq(10, 2, 10, 3));
	seq1->setIntervalForDim(1, new IntervalSeq(0, 5, 100, 5));
	list1->Append(seq1);

	MultidimensionalIntervalSeq *seq2 = new MultidimensionalIntervalSeq(1);
	seq2->setIntervalForDim(0, new IntervalSeq(5, 2, 25, 4));
	list1->Append(seq2);

	MultidimensionalIntervalSeq *seq3 = new MultidimensionalIntervalSeq(3);
	seq3->setIntervalForDim(0, new IntervalSeq(100, 1, 10, 10));
	seq3->setIntervalForDim(1, new IntervalSeq(50, 25, 50, 1));
	seq3->setIntervalForDim(2, new IntervalSeq(0, 10, 25, 2));
	list1->Append(seq3);

	const char *strDesc = MultidimensionalIntervalSeq::convertSetToString(list1);
	cout << strDesc << "\n";

	List<MultidimensionalIntervalSeq*> *list2 = MultidimensionalIntervalSeq::constructSetFromString(strDesc);
	for (int i = 0; i < list2->NumElements(); i++) {
		cout << "Sequence #" << i << "\n";
		list2->Nth(i)->draw();
	}

	return 0;
}


