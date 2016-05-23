#include "list.h"
#include "structures.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <deque>

#include "utils.h"

using namespace std;

template<class Type> class FileReader {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;
	int elementsCount;

public:
	FileReader(const char *fileName) {
		this->fileName = fileName;
		this->dimLengths = new List<Dimension*>;
		this->elementsCount = 0;
	}

	Type *readArray() {

		std::ifstream file(fileName);
		if (!file.is_open()) {
			cout << "could not open input file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}

		string input;
		getline(file, input);
		string delim = "*";
		std::deque<string> tokenList = tokenizeString(input, delim);
		while (!tokenList.empty()) {
			string token = tokenList.front();
			tokenList.pop_front();
			Dimension *dimension = new Dimension;
			trim(token);
			dimension->length = atoi(token.c_str());
			dimension->range.min = 0;
			dimension->range.max = dimension->length - 1;
			dimLengths->Append(dimension);
		}

		elementsCount = 1;
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			elementsCount *= dimLengths->Nth(i)->length;
		}

		Type *array = new Type[elementsCount];
		int readCount = 0;
		Type nextElement;
		while (file >> nextElement) {
			array[readCount] = nextElement;
			readCount++;
		}
		if (readCount < elementsCount) {
			cout << "specified file does not have enough data elements: ";
			cout << "read only " << readCount << " values\n";
			exit(EXIT_FAILURE);
		}

		file.close();
		return array;
	}

	List<Dimension*> *getDimLengths() { return dimLengths; }
	int getElementsCount() { return elementsCount; }
};

int mainDFC(int argc, char *argv[]) {

	const char *file1Name = NULL;
	if (argc > 1) {
		file1Name = argv[1];
	} else {
		cout << "input file(s) unspecified: sequence is file1 file2 data-type\n";
		cout << "as a fourth argument you can specify the tolerance level of difference (default is 0)\n";
		exit(EXIT_FAILURE);
	}
	const char *file2Name = NULL;
	if (argc > 2) {
		file2Name = argv[2];
	} else {
		cout << "input file(s) unspecified: sequence is file1 file2 data-type\n";
		cout << "as a fourth argument you can specify the tolerance level of difference (default is 0)\n";
		exit(EXIT_FAILURE);
	}
	const char *dataType = NULL;
	if (argc > 3) {
		dataType = argv[3];
	} else {
		cout << "data type (char, int, float, or double) unspecified: sequence is file1 file2 data-type\n";
		cout << "as a fourth argument you can specify the tolerance level of difference (default is 0)\n";
		exit(EXIT_FAILURE);
	}
	double toleranceLevel = 0;
	if (argc > 4) {
		istringstream istream(argv[4]);
		istream >> toleranceLevel;
	}


	if (strcmp("double", dataType) == 0) {
		FileReader<double> *reader1 = new FileReader<double>(file1Name);
		double *array1 = reader1->readArray();
		FileReader<double> *reader2 = new FileReader<double>(file2Name);
		double *array2 = reader2->readArray();
		int array1Length = reader1->getElementsCount();
		int array2Length = reader2->getElementsCount();
		if (array1Length != array2Length) {
			cout << "two files have different numbers of elements\n";
			cout << array1Length << "-" << array2Length << "\n";
			exit(EXIT_FAILURE);
		}
		int mismatchCount = 0;
		for (int i = 0; i < array1Length; i++) {
			if (abs(array1[i] - array2[i]) > toleranceLevel) {
				cout << "Mismatch: first " << array1[i] << " second " << array2[i] << "\n";
				mismatchCount++;
			}
		}
		if (mismatchCount > 0) {
			cout << "Two files differ in " << mismatchCount << " data points\n";
		} else {
			cout << "Two files contain same data\n";
		}
	} else if (strcmp("float", dataType) == 0) {
		FileReader<float> *reader1 = new FileReader<float>(file1Name);
		float *array1 = reader1->readArray();
		FileReader<float> *reader2 = new FileReader<float>(file2Name);
		float *array2 = reader2->readArray();
		int array1Length = reader1->getElementsCount();
		int array2Length = reader2->getElementsCount();
		if (array1Length != array2Length) {
			cout << "two files have different numbers of elements\n";
			cout << array1Length << "-" << array2Length << "\n";
			exit(EXIT_FAILURE);
		}
		int mismatchCount = 0;
		for (int i = 0; i < array1Length; i++) {
			if (abs(array1[i] - array2[i]) > toleranceLevel) {
				cout << "Mismatch: first " << array1[i] << " second " << array2[i] << "\n";
				mismatchCount++;
			}
		}
		if (mismatchCount > 0) {
			cout << "Two files differ in " << mismatchCount << " data points\n";
		} else {
			cout << "Two files contain the same data\n";
		}
	} else if (strcmp("int", dataType) == 0) {
		FileReader<int> *reader1 = new FileReader<int>(file1Name);
		int *array1 = reader1->readArray();
		FileReader<int> *reader2 = new FileReader<int>(file2Name);
		int *array2 = reader2->readArray();
		int array1Length = reader1->getElementsCount();
		int array2Length = reader2->getElementsCount();
		if (array1Length != array2Length) {
			cout << "two files have different numbers of elements\n";
			exit(EXIT_FAILURE);
		}
		int mismatchCount = 0;
		for (int i = 0; i < array1Length; i++) {
			if (abs(array1[i] - array2[i]) > toleranceLevel) {
				cout << "Mismatch: first " << array1[i] << " second " << array2[i] << "\n";
				mismatchCount++;
			}
		}
		if (mismatchCount > 0) {
			cout << "Two files differ in " << mismatchCount << " data points\n";
		} else {
			cout << "Two files contain same data\n";
		}
	} else if (strcmp("char", dataType) == 0) {
		FileReader<char> *reader1 = new FileReader<char>(file1Name);
		char *array1 = reader1->readArray();
		FileReader<char> *reader2 = new FileReader<char>(file2Name);
		char *array2 = reader2->readArray();
		int array1Length = reader1->getElementsCount();
		int array2Length = reader2->getElementsCount();
		if (array1Length != array2Length) {
			cout << "two files have different numbers of elements\n";
			exit(EXIT_FAILURE);
		}
		int mismatchCount = 0;
		for (int i = 0; i < array1Length; i++) {
			if (abs(array1[i] - array2[i]) > toleranceLevel) {
				cout << "Mismatch: first " << array1[i] << " second " << array2[i] << "\n";
				mismatchCount++;
			}
		}
		if (mismatchCount > 0) {
			cout << "Two files differ in " << mismatchCount << " data points\n";
		} else {
			cout << "Two files contain same data\n";
		}
	}

	return 0;
}
