#include "list.h"
#include "structures.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>

#include "utils.h"

using namespace std;

template<class Type> class ArrayGenerator {
private:
	Type multiplier;
public:
	void init(Type multiplier) {
		this->multiplier = multiplier;
		srand(time(NULL));
	}
	Type generateItem(int min, int max) {
		return std::max(rand() % max * multiplier, min * multiplier);
	}
};

int mainAG(int argc, char *argv[]) {

	ostringstream message;
	message << "To use this array generator, you have to provide 5 command line arguments\n";
	message << "First specify the dimensionality of the array in following format: ";
	message << "dim1Length*dim2Length*...\n";
	message << "For example if you want to a 10-by-20-by-50 cubic array then the input should be 10*20*50\n";
	message << "The second argument specifies the type of the array elements\n";
	message << "Valid values are char, int, float, and double\n";
	message << "The next two arguments specify the minimum and maximum values for array elements\n";
	message << "The last argument specifies the file name you want the generated array to be written into\n";
	message << "So an example command line sequence can be as follows\n";
	message << "100*100 float 0.25 10000 input_matrix.txt\n";

	if (argc < 6) {
		cout << message.str();
		exit(EXIT_FAILURE);
	}

	string dimensionStr = string(argv[1]);
	string delim = "*";
	std::deque<string> tokenList = tokenizeString(dimensionStr, delim);
	int elementCount = 1;
	int lineBreakAtElement = 0;
	while (!tokenList.empty()) {
		string token = tokenList.front();
		tokenList.pop_front();
		elementCount *= atoi(token.c_str());
		lineBreakAtElement = atoi(token.c_str());
	}

	const char *dataType = argv[2];

	double actualMin = atof(argv[3]);
	int minInt = (int) actualMin;
	double actualMax = atof(argv[4]);
	int maxInt = (int) actualMax;

	const char *outputFile = argv[5];
	ofstream stream;
	stream.open(outputFile);
	if (!stream.is_open()) {
		cout << "could not open output file: " << outputFile << "\n";
		exit(EXIT_FAILURE);
	}
	stream << dimensionStr << "\n";
	if (strcmp("double", dataType) == 0) {
		ArrayGenerator<double> *generator = new ArrayGenerator<double>;
		generator->init(1.0101);
		stream << generator->generateItem(minInt, maxInt);
		for (int i = 1; i < elementCount; i++) {
			if (i % lineBreakAtElement == 0) {
				stream << '\n';
			} else stream << ' ';
			stream << generator->generateItem(minInt, maxInt);
		}
	} else if (strcmp("float", dataType) == 0) {
		ArrayGenerator<float> *generator = new ArrayGenerator<float>;
		generator->init(1.01f);
		stream << generator->generateItem(minInt, maxInt);
		for (int i = 1; i < elementCount; i++) {
			if (i % lineBreakAtElement == 0) {
				stream << '\n';
			} else stream << ' ';
			stream << generator->generateItem(minInt, maxInt);
		}
	} else if (strcmp("int", dataType) == 0) {
		ArrayGenerator<int> *generator = new ArrayGenerator<int>;
		generator->init(1);
		stream << generator->generateItem(minInt, maxInt);
		for (int i = 1; i < elementCount; i++) {
			if (i % lineBreakAtElement == 0) {
				stream << '\n';
			} else stream << ' ';
			stream << generator->generateItem(minInt, maxInt);
		}
	} else if (strcmp("char", dataType) == 0) {
		ArrayGenerator<char> *generator = new ArrayGenerator<char>;
		generator->init(1);
		stream << generator->generateItem(minInt, maxInt);
		for (int i = 1; i < elementCount; i++) {
			if (i % lineBreakAtElement == 0) {
				stream << '\n';
			} else stream << ' ';
			stream << generator->generateItem(minInt, maxInt);
		}
	}

	stream.close();
	cout << "wrote generated array to output file: " << outputFile << "\n";
	return 0;
}
