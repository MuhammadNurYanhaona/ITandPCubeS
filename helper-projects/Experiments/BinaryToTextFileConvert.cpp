#include "list.h"
#include "structures.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>

#include "utils.h"

using namespace std;

template<class Type> class BinaryStreamReader {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;
	List<int> *dimMultiplier;
	int dataBegins;
	int seekStepSize;
	ifstream stream;

public:
	BinaryStreamReader(const char *fileName) {
		this->fileName = fileName;
		seekStepSize = sizeof(Type);
		initialize();
	}

	void open() {
		stream.open(fileName, ios_base::binary);
		if (!stream.is_open()) {
			cout << "could not open input file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}
		stream.seekg(dataBegins, ios_base::beg);
	}
	void close() { stream.close(); }

	Type readElement(List<int> *index) {
		long int seekPosition = getSeekPosition(index);
		stream.seekg(seekPosition, ios_base::beg);
		Type element;
		stream.read(reinterpret_cast<char*>(&element), seekStepSize);
		return element;
	}

	List<Dimension*> *getDimensions() { return dimLengths; }

	Type *readArray() {
		open();
		int size = 1;
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			size *= dimLengths->Nth(i)->length;
		}
		Type *array = new Type[size];
		List<int> *elementIndex = new List<int>;
		readElements(array, 0, elementIndex);
		close();
		return array;
	}

private:

	void initialize() {

		// try to open the file and if failed exit with an error
		stream.open(fileName, ifstream::binary);
		if (!stream.is_open()) {
			cout << "could not open input file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}

		// read dimension information from the beginning of the file
		List<char> *dimInfo = new List<char>;
		char *ch = new char;
		do {
			stream.read(ch, sizeof(char));
			dimInfo->Append(*ch);
		} while (*ch != '\n');
		char *dimInfoBuffer = new char[dimInfo->NumElements()];
		for (int i = 0; i < dimInfo->NumElements() - 1; i++) {
			dimInfoBuffer[i] = dimInfo->Nth(i);
		}
		dimInfoBuffer[dimInfo->NumElements()] = '\0';

		// get the individual dimensions from the dimension information string
		dimLengths = new List<Dimension*>;
		string str(dimInfoBuffer);
		string delim = "*";
		deque<string> tokenList = tokenizeString(str, delim);
		while (!tokenList.empty()) {
			string token = tokenList.front();
			tokenList.pop_front();
			istringstream str(token);
			int dimensionLength;
			str >> dimensionLength;
			Dimension *dim = new Dimension;
			dim->length = dimensionLength;
			dim->range.min = 0;
			dim->range.max = dimensionLength - 1;
			dimLengths->Append(dim);
		}

		// set the beginning of data section pointer appropriately to be used later when reading elements
		dataBegins = stream.tellg();

		// initialize the dim-multiplier-list for random access
		dimMultiplier = new List<int>;
		int currentMultiplier = 1;
		for (int i = dimLengths->NumElements() - 1; i >= 0; i--) {
			dimMultiplier->InsertAt(currentMultiplier, 0);
			currentMultiplier *= dimLengths->Nth(i)->length;
		}

		stream.close();
	}

	long int getSeekPosition(List<int> *elementIndex) {
		long int positionNo = 0;
		for (int i = 0; i < elementIndex->NumElements(); i++) {
			positionNo += elementIndex->Nth(i) * dimMultiplier->Nth(i);
		}
		return (positionNo * seekStepSize + dataBegins);
	}

	long int getStoreIndex(List<int> *elementIndex) {
		long int index = 0;
		for (int i = 0; i < elementIndex->NumElements(); i++) {
			index += elementIndex->Nth(i) * dimMultiplier->Nth(i);
		}
		return index;
	}

	void readElements(Type *array, int currentDimNo, List<int> *partialIndex) {
	        Dimension *dimension = dimLengths->Nth(currentDimNo);
	        for (int index = dimension->range.min; index <= dimension->range.max; index++) {
	                partialIndex->Append(index);
	                if (currentDimNo < dimLengths->NumElements() - 1) {
	                        readElements(array, currentDimNo + 1, partialIndex);
	                } else {
	                		array[getStoreIndex(partialIndex)] = readElement(partialIndex);
	                }
	                partialIndex->RemoveAt(currentDimNo);
	        }
	}
};

template<class Type> class TextStreamWriter {
protected:
	List<Dimension*> *dimLengths;
	const char *fileName;

public:
	TextStreamWriter(List<Dimension*> *dimLengths, const char *fileName) {
		this->dimLengths = dimLengths;
		this->fileName = fileName;
	}

	void writeArray(Type *array) {
		ofstream file(fileName);
		if (!file.is_open()) {
			cout << "could not open output file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}

		int elementsCount = 1;
		int lastDimLength = 0;
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			if (i > 0) file << " * ";
			lastDimLength = dimLengths->Nth(i)->length;
			elementsCount *= lastDimLength;
			file << lastDimLength;
		}

		file << '\n';
		int outputOnALine = 0;
		for (int i = 0; i < elementsCount; i++) {
			file << array[i];
			outputOnALine++;
			if (outputOnALine < lastDimLength) {
				file << ' ';
			} else {
				file << '\n';
				outputOnALine = 0;
			}
		}
		file.close();
	}
};

int mainBinToText(int argc, char *argv[]) {

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
		BinaryStreamReader<double> *reader = new BinaryStreamReader<double>(inputFileName);
		double *array = reader->readArray();
		List<Dimension*> *arrayDims = reader->getDimensions();
		cout << "read input file: " << inputFileName << "\n";
		TextStreamWriter<double> *writer = new TextStreamWriter<double>(arrayDims, outputFileName);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("float", dataType) == 0) {
		BinaryStreamReader<float> *reader = new BinaryStreamReader<float>(inputFileName);
		float *array = reader->readArray();
		List<Dimension*> *arrayDims = reader->getDimensions();
		cout << "read input file: " << inputFileName << "\n";
		TextStreamWriter<float> *writer = new TextStreamWriter<float>(arrayDims, outputFileName);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("int", dataType) == 0) {
		BinaryStreamReader<int> *reader = new BinaryStreamReader<int>(inputFileName);
		int *array = reader->readArray();
		List<Dimension*> *arrayDims = reader->getDimensions();
		cout << "read input file: " << inputFileName << "\n";
		TextStreamWriter<int> *writer = new TextStreamWriter<int>(arrayDims, outputFileName);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	} else if (strcmp("char", dataType) == 0) {
		BinaryStreamReader<char> *reader = new BinaryStreamReader<char>(inputFileName);
		char *array = reader->readArray();
		List<Dimension*> *arrayDims = reader->getDimensions();
		cout << "read input file: " << inputFileName << "\n";
		TextStreamWriter<char> *writer = new TextStreamWriter<char>(arrayDims, outputFileName);
		writer->writeArray(array);
		cout << "wrote to output file: " << outputFileName << "\n";
	}

	return 0;
}



