#include <iostream>
#include <vector>
#include <cstdlib>
#include "../utils/interval.h"
#include "../utils/list.h"
#include "../communication/confinement_mgmt.h"

using namespace std;

int mainITT() {

	// an example 1D sequence
	MultidimensionalIntervalSeq *seq1 = new MultidimensionalIntervalSeq(1);
	seq1->setIntervalForDim(0, new IntervalSeq(0, 5, 20, 3));

	// an example 2D sequence
	MultidimensionalIntervalSeq *seq2 = new MultidimensionalIntervalSeq(2);
	seq2->setIntervalForDim(0, new IntervalSeq(0, 2, 10, 2));
	seq2->setIntervalForDim(1, new IntervalSeq(15, 1, 20, 5));

	// an example 3D sequence
	MultidimensionalIntervalSeq *seq3 = new MultidimensionalIntervalSeq(3);
	seq3->setIntervalForDim(0, new IntervalSeq(0, 2, 10, 2));
	seq3->setIntervalForDim(1, new IntervalSeq(10, 2, 20, 2));
	seq3->setIntervalForDim(2, new IntervalSeq(100, 2, 100, 3));

	// an example of a data exchange list
	MultidimensionalIntervalSeq *seq4 = new MultidimensionalIntervalSeq(2);
	seq4->setIntervalForDim(0, new IntervalSeq(0, 2, 10, 2));
	seq4->setIntervalForDim(1, new IntervalSeq(0, 1, 10, 2));
	MultidimensionalIntervalSeq *seq5 = new MultidimensionalIntervalSeq(2);
	seq5->setIntervalForDim(0, new IntervalSeq(20, 2, 10, 2));
	seq5->setIntervalForDim(1, new IntervalSeq(20, 2, 10, 2));
	MultidimensionalIntervalSeq *seq6 = new MultidimensionalIntervalSeq(2);
	seq6->setIntervalForDim(0, new IntervalSeq(15, 2, 20, 1));
	seq6->setIntervalForDim(1, new IntervalSeq(15, 1, 20, 1));
	List<MultidimensionalIntervalSeq*> *seqList = new List<MultidimensionalIntervalSeq*>;
	seqList->Append(seq4);
	seqList->Append(seq5);
	seqList->Append(seq6);


	cout << "Iterator for a single sequence:\n";
	SequenceIterator *iterator = new SequenceIterator(seq3);
	while (iterator->hasMoreElements()) {
		iterator->printNextElement(cout);
	}

	cout << "\nIterator for a a list of sequences:\n";
	DataExchange *exchange = new DataExchange(0, 1, seqList);
	ExchangeIterator *listIterator = new ExchangeIterator(exchange);
	while (listIterator->hasMoreElements()) {
		listIterator->printNextElement(cout);
	}

	return 0;
}



