#include "../utils/list.h"
#include "../utils/interval.h"
#include "../utils/string_utils.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>

using namespace std;

bool validateIntersection(MultidimensionalIntervalSeq *seq1,
			MultidimensionalIntervalSeq *seq2,
			List<MultidimensionalIntervalSeq*> *intersect) {

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
		cout << "intersection calculation error!!!\n";
		cout << "matching found in " << commonElements << " places but the intersection has ";
		cout << intersectElements << " elements\n";
	}

	return commonElements == intersectElements;
}

int mainSIIT(int argc, char *argv[]) {

	string delim(" ");

	while (true) {

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

		bool valid = true;
		List<MultidimensionalIntervalSeq*> *overlap = new List<MultidimensionalIntervalSeq*>;
		for (int i = 0; i < firstSet->NumElements(); i++) {
			MultidimensionalIntervalSeq *seq1 = firstSet->Nth(i);
			for (int j = 0; j < secondSet->NumElements(); j++) {
				MultidimensionalIntervalSeq *seq2 = secondSet->Nth(j);
				List<MultidimensionalIntervalSeq*> *intersection = seq1->computeIntersection(seq2);
				valid = valid && validateIntersection(seq1, seq2, intersection);
				if (intersection != NULL) {
					overlap->AppendAll(intersection);
				}
			}
		}
		if (valid) {
			cout << "interval calculation was correct :)\n";
		} else {
			cout << "interval calculation was incorrect!!! :(\n";
		}
		delete list;
	}
	cout << "Program terminated\n";

	return 0;
}


