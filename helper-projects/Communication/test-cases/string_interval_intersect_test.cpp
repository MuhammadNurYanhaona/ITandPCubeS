#include "../utils/list.h"
#include "../utils/interval.h"
#include "../utils/string_utils.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>

using namespace std;

void printErrorCase(Dimension dim, IntervalSeq *interval1, IntervalSeq *interval2) {
	List<IntervalSeq*> *intersection = interval1->computeIntersection(interval2, false);
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
}

bool validateIntersection(MultidimensionalIntervalSeq *seq1, MultidimensionalIntervalSeq *seq2,
			List<MultidimensionalIntervalSeq*> *intersect, Dimension dim) {

	SequenceIterator *seq1Iterator = new SequenceIterator(seq1);
	int commonElements = 0;
	while (seq1Iterator->hasMoreElements()) {
		vector<int> *index1 = seq1Iterator->getNextElement();
		SequenceIterator *seq2Iterator = new SequenceIterator(seq2);
		while (seq2Iterator->hasMoreElements()) {
			vector<int> *index2 = seq2Iterator->getNextElement();

			bool mismatch = false;
			for (unsigned int i = 0; i < index1->size(); i++) {
				if (index1->at(i) != index2->at(i)) {
					mismatch = true;
					break;
				}
			}
			if (!mismatch) {
				commonElements++;
				break;
			}
		}
		delete seq2Iterator;
	}
	delete seq1Iterator;

	int intersectElements = 0;
	if (intersect != NULL) {
		for (int i = 0; i < intersect->NumElements(); i++) {
			intersectElements += intersect->Nth(i)->getNumOfElements();
		}
	}

	if (commonElements != intersectElements) {
		cout << "intersection calculation error!!! in ";
		cout << seq1->toString() << " " << seq2->toString();
		cout << " matching found in " << commonElements << " places but the intersection has ";
		cout << intersectElements << " elements\n";
		for (int i = 0; i < seq1->getDimensionality(); i++) {
			printErrorCase(dim, seq1->getIntervalForDim(i), seq2->getIntervalForDim(i));
		}
	}

	return commonElements == intersectElements;
}

int mainSIIT(int argc, char *argv[]) {

	string delim(" ");
	bool printingEnabled = false;

	while (true) {

		Dimension dim = Dimension(50);
		cout << "Enter 'quit' to terminate\n";
		string str;
		cout << "Enter the two string interval descriptions separated by blank space\n";
		getline(cin, str);
		if (strcmp(str.c_str(), "quit") == 0) break;

		List<string> *list = string_utils::tokenizeString(str, delim);
		string str1 = list->Nth(0);
		string str2 = list->Nth(1);

		List<MultidimensionalIntervalSeq*> *firstSet = MultidimensionalIntervalSeq::constructSetFromString(str1.c_str());
		List<MultidimensionalIntervalSeq*> *secondSet = MultidimensionalIntervalSeq::constructSetFromString(str2.c_str());
		if (printingEnabled) {
			cout << "\nFirst Sequence:---------------------------\n";
			for (int i = 0; i < firstSet->NumElements(); i++) {
				firstSet->Nth(i)->draw(dim);
			}
			cout << "\nSecond Sequence:---------------------------\n";
			for (int i = 0; i < secondSet->NumElements(); i++) {
				secondSet->Nth(i)->draw(dim);
			}
		}

		bool valid = true;
		List<MultidimensionalIntervalSeq*> *overlap = new List<MultidimensionalIntervalSeq*>;
		for (int i = 0; i < firstSet->NumElements(); i++) {
			MultidimensionalIntervalSeq *seq1 = firstSet->Nth(i);
			for (int j = 0; j < secondSet->NumElements(); j++) {
				MultidimensionalIntervalSeq *seq2 = secondSet->Nth(j);
				List<MultidimensionalIntervalSeq*> *intersection = seq1->computeIntersection(seq2);
				valid = valid && validateIntersection(seq1, seq2, intersection, dim);
				if (intersection != NULL) {
					overlap->AppendAll(intersection);
				}
			}
		}
		if (valid) {
			cout << "interval calculation was correct :)\n\n\n\n";
		} else {
			cout << "interval calculation was incorrect!!! :(\n\n\n\n";
		}
		delete list;
	}
	cout << "Program terminated\n";

	return 0;
}


