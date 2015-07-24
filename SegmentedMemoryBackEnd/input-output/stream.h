#ifndef _H_stream
#define _H_stream

/* This header file defines two stream classes for reading and writing arrays to and from binary files. This 
   classes are templated; so any primitive type for elements of an array is supported naturally. To support 
   arrays of objects, minor changes be required (date: Jun-15-2015). In particular, regarding zero filling the 
   output stream at the beginning. 
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>

#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../codegen/structure.h"

using namespace std;

template <class Type> class TypedInputStream {
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
	~TypedInputStream() {
		while (dimLengths->NumElements() > 0) {
			Dimension *dimension = dimLengths->Nth(0);
			dimLengths->RemoveAt(0);
			delete dimension;
		} 
		delete dimLengths;
		delete dimMultiplier; 
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

	// read an element at a specific index of the array 
	Type readElement(List<int> *index) {
		long int seekPosition = getSeekPosition(index);
		stream.seekg(seekPosition, ios_base::beg);
		Type element;
		stream.read(reinterpret_cast<char*>(&element), seekStepSize);
		return element;
	}

	// read the element from current file read pointer location; use this with care 
	Type readNextElement() {
		Type element;
		stream.read(reinterpret_cast<char*>(&element), seekStepSize);
		return element;
	}

	void copyDimensionInfo(Dimension *dimension) {
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			dimension[i] = *(dimLengths->Nth(i));
		}
	}

	void copyDimensionInfo(PartDimension *partDims) {
		for (int i = 0; i < dimLengths->NumElements(); i++) {
			partDims[i].partition = *(dimLengths->Nth(i));
		}
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
		char *ch = new char[8];
		do {
			stream.read(ch, sizeof(char));
			dimInfo->Append(*ch);
		} while (*ch != '\n');
		char *dimInfoBuffer = new char[dimInfo->NumElements()];
		for (int i = 0; i < dimInfo->NumElements() - 1; i++) {
			dimInfoBuffer[i] = dimInfo->Nth(i);
		}
		dimInfoBuffer[dimInfo->NumElements() - 1] = '\0';
		delete dimInfo;

		// get the individual dimensions from the dimension information string
		dimLengths = new List<Dimension*>;
		string str(dimInfoBuffer);
		string delim = "*";
		List<string> *tokenList = string_utils::tokenizeString(str, delim);
		while (tokenList->NumElements() > 0) {
			string token = tokenList->Nth(0);
			tokenList->RemoveAt(0);
			istringstream str(token);
			int dimensionLength;
			str >> dimensionLength;
			Dimension *dim = new Dimension;
			dim->range.min = 0;
			dim->range.max = dimensionLength - 1;
			dim->setLength();
			dimLengths->Append(dim);
		}
		delete[] dimInfoBuffer;
		delete tokenList;

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
		delete[] ch;
	}

	long int getSeekPosition(List<int> *elementIndex) {
		long int positionNo = 0;
		for (int i = 0; i < elementIndex->NumElements(); i++) {
			positionNo += elementIndex->Nth(i) * dimMultiplier->Nth(i);
		}
		return (positionNo * seekStepSize + dataBegins);
	}
};

template <class Type> class TypedOutputStream {
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
	~TypedOutputStream() {
		delete dimLengths;
		delete dimMultiplier; 
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

#endif
