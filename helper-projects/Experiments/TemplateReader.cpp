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

template<class Type> class TypedInputStream {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;
	List<int> *dimMultiplier;
	int dataBegins;
	int seekStepSize;
	ifstream stream;
public:
	TypedInputStream(const char *fileName) {
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

};

template<class Type> class TypedOutputStream {
protected:
	const char *fileName;
	List<Dimension*> *dimLengths;
	List<int> *dimMultiplier;
	int dataBegins;
	int seekStepSize;
	ofstream stream;

public:
	TypedOutputStream(const char *fileName, List<Dimension*> *dimLengths, bool initFile) {
		this->dimLengths = dimLengths;
		this->fileName = fileName;
		seekStepSize = sizeof(Type);
		if (initFile) initializeFile();
		initialize();
	}

	void open() {
		// note that the file has to be opened in read-write mode; otherwise overwriting cannot be done on specific points
		stream.open(fileName, ios_base::binary | ios_base::in | ios_base::out);
		if (!stream.is_open()) {
			cout << "could not open output file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}
		stream.seekp(dataBegins, ios_base::beg);
	}
	void close() { stream.close(); }

	void writeElement(Type element, List<int> *index) {
		long int seekPosition = getSeekPosition(index);
		stream.seekp(seekPosition, ios_base::beg);
		stream.write(reinterpret_cast<char*>(&element), seekStepSize);
	}

	// write elements at the current location of the seek pointer; use this with care
	void writeNextElement(Type element) {
		stream.write(reinterpret_cast<char*>(&element), seekStepSize);
	}

private:
	void initializeFile() {

		// try to open the file and exit the program if failed
		stream.open(fileName, ios_base::binary);
		if (!stream.is_open()) {
			cout << "could not open output file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}

		// write the dimension length information at the beginning of the file
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

		// zero fill the file to facilitate later update without facing the problem of crossing the end-of-file marker
		Type zero = 0;
		for (int i = 0; i < totalElements; i++) {
			stream.write(reinterpret_cast<char*>(&zero), seekStepSize);
		}
		stream.close();
	}

	void initialize() {

		// try to open the file and exit the program if failed
		ifstream istream(fileName, ios_base::binary);
		if (!istream.is_open()) {
			cout << "could not open output file: " << fileName << "\n";
			exit(EXIT_FAILURE);
		}

		// find out where the dimension information has ended and data writing should being
		char ch;
		do {
			istream.read(&ch, sizeof(char));
		} while (ch != '\n');

		// set up the seek pointer's beginning to determine proper places to fill in elements from data parts
		dataBegins = istream.tellg();
		istream.close();

		// initialize the dim-multiplier-list for later random access updates
		dimMultiplier = new List<int>;
		int currentMultiplier = 1;
		for (int i = dimLengths->NumElements() - 1; i >= 0; i--) {
			dimMultiplier->InsertAt(currentMultiplier, 0);
			currentMultiplier *= dimLengths->Nth(i)->length;
		}
	}

	long int getSeekPosition(List<int> *elementIndex) {
		long int positionNo = 0;
		for (int i = 0; i < elementIndex->NumElements(); i++) {
			positionNo += elementIndex->Nth(i) * dimMultiplier->Nth(i);
		}
		return (positionNo * seekStepSize + dataBegins);
	}
};

int mainTemRead() {

	List<Dimension*> *dims = new List<Dimension*>;
	Dimension *rows = new Dimension;
	rows->length = 10;
	dims->Append(rows);
	Dimension *cols = new Dimension;
	cols->length = 5;
	dims->Append(cols);
	const char* fileName = "/home/yan/file";
	TypedOutputStream<double> *ostream = new TypedOutputStream<double>(fileName, dims, true);
	ostream->open();
	for (int i = 9; i >= 0; i--) {
		for (int j = 0; j < 5; j++) {
			List<int> *index = new List<int>;
			index->Append(i);
			index->Append(j);
			double data = (i + j) * 1.01;
			ostream->writeElement(data, index);
		}
	}
	ostream->close();

	TypedOutputStream<double> *ostream2 = new TypedOutputStream<double>(fileName, dims, false);
	ostream2->open();
	for (int i = 3; i <= 6; i++) {
		for (int j = 1; j < 4; j++) {
			List<int> *index = new List<int>;
			index->Append(i);
			index->Append(j);
			double data = 1.0;
			ostream2->writeElement(data, index);
		}
	}
	ostream2->close();

	double data[10][5];
	TypedInputStream<double> *istream = new TypedInputStream<double>("/home/yan/file");
	istream->open();
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 10; i++) {
			List<int> *index = new List<int>;
			index->Append(i);
			index->Append(j);
			double f = istream->readElement(index);
			data[i][j] = f;
		}
	}
	istream->close();
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 5; j++) {
			cout << data[i][j] << "\t";
		}
		cout << "\n";
	}

	return 0;
}
