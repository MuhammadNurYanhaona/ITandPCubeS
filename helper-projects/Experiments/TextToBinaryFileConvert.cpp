#include "list.h"
#include "structures.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <deque>

#include "utils.h"

using namespace std;

template<class Type> class TextFileReader {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;

public:
	TextFileReader(const char *fileName) {
		this->fileName = fileName;
		this->dimLengths = new List<Dimension*>;
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

		int elementsCount = 1;
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
};

template<class Type> class BinaryFileWriter {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;
	ofstream stream;

public:
	BinaryFileWriter(const char *fileName, List<Dimension*> *dimLengths) {
		this->dimLengths = dimLengths;
		this->fileName = fileName;
	}

	void open() {
		stream.open(fileName, ios_base::binary);
		if (!stream.is_open()) {
			cout << "could not open output file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}
	}
	void close() { stream.close(); }

	void writeArray(Type *array) {

		open();

		long int totalElements = 1;
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			ostringstream str;
			if (i > 0) str << "*";
			int dimLength = dimLengths->Nth(i)->length;
			str << dimLength;
			totalElements *= dimLength;
			int strLength = str.str().length();
			stream.write(str.str().c_str(), sizeof(char) * strLength);
			stream.flush();
		}
		char lineEnd = '\n';
		stream.write(&lineEnd, sizeof(char));

		for (int i = 0; i < totalElements; i++) {
			Type element = array[i];
			stream.write(reinterpret_cast<char*>(&element), sizeof(Type));
		}

		close();
	}
};

int mainTextToBin(int argc, char *argv[]) {

	const char *inputFileName = NULL;
	if (argc > 1) {
		inputFileName = argv[1];
	} else {
		cout << "input file unspecified: sequence is input-file data-type output-file\n";
		exit(EXIT_FAILURE);
	}
	const char *dataType = NULL;
	if (argc > 2) {
		dataType = argv[2];
	} else {
		cout << "input file's data type (char, int, float, or double) unspecified:  sequence is input-file data-type output-file\n";
		exit(EXIT_FAILURE);
	}
	const char *outputFileName = NULL;
	if (argc > 3) {
		outputFileName = argv[3];
	} else {
		cout << "output file unspecified: sequence is input-file data-type output-file\n";
		exit(EXIT_FAILURE);
	}

	if (strcmp("double", dataType) == 0) {
		TextFileReader<double> *reader = new TextFileReader<double>(inputFileName);
		double *array = reader->readArray();
		List<Dimension*> *dimensionList = reader->getDimLengths();
		cout << "read input file: " << inputFileName << "\n";
		BinaryFileWriter<double> *writer = new BinaryFileWriter<double>(outputFileName, dimensionList);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("float", dataType) == 0) {
		TextFileReader<float> *reader = new TextFileReader<float>(inputFileName);
		float *array = reader->readArray();
		List<Dimension*> *dimensionList = reader->getDimLengths();
		cout << "read input file: " << inputFileName << "\n";
		BinaryFileWriter<float> *writer = new BinaryFileWriter<float>(outputFileName, dimensionList);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("int", dataType) == 0) {
		TextFileReader<int> *reader = new TextFileReader<int>(inputFileName);
		int *array = reader->readArray();
		List<Dimension*> *dimensionList = reader->getDimLengths();
		cout << "read input file: " << inputFileName << "\n";
		BinaryFileWriter<int> *writer = new BinaryFileWriter<int>(outputFileName, dimensionList);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("char", dataType) == 0) {
		TextFileReader<char> *reader = new TextFileReader<char>(inputFileName);
		char *array = reader->readArray();
		List<Dimension*> *dimensionList = reader->getDimLengths();
		cout << "read input file: " << inputFileName << "\n";
		BinaryFileWriter<char> *writer = new BinaryFileWriter<char>(outputFileName, dimensionList);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	}

	return 0;
}
