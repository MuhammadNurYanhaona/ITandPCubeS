#include "../utils/list.h"
#include "../utils/interval.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

int mainPIIT() {

	MultidimensionalIntervalSeq *seq = new MultidimensionalIntervalSeq(2);
	IntervalSeq *interval1 = new IntervalSeq(5, 3, 10, 3);
	seq->setIntervalForDim(0, interval1);
	IntervalSeq *interval2 = new IntervalSeq(15, 1, 5, 10);
	seq->setIntervalForDim(1, interval2);
	seq->draw();
	cout << "\n";

	List<int> *point = new List<int>;

	// test 1
	point->Append(4); point->Append(15);
	cout << "contains (4, 15): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 2
	point->Append(16); point->Append(20);
	cout << "contains (16, 20): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 3
	point->Append(16); point->Append(65);
	cout << "contains (16, 65): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 4
	point->Append(27); point->Append(60);
	cout << "contains (27, 60): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 5
	point->Append(28); point->Append(55);
	cout << "contains (28, 55): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 6
	point->Append(29); point->Append(59);
	cout << "contains (29, 59): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	// test 7
	point->Append(25); point->Append(50);
	cout << "contains (25, 50): " << (seq->contains(point) ? "True" : "False") << "\n";
	point->clear();

	return 0;
}
