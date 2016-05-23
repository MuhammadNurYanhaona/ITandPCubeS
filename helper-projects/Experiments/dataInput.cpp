#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include "utils.h"
#include "structures.h"

void readArrayDimensionInfoFromFile(std::ifstream &file, int dimensionCount, Dimension *dimensions) {
	std::string input;
	std::getline(file, input);
	std::string delim = "*";
	std::deque<std::string> dimensionList = tokenizeString(input, delim);
	for (int i = 0; i < dimensionCount; i++) {
		std::string token = dimensionList.front();
		dimensions[i].range.min = 0;
		trim(token);
		dimensions[i].range.max = atoi(token.c_str()) - 1;
		dimensionList.pop_front();
		dimensions[i].length = abs(dimensions[i].range.max - dimensions[i].range.min) + 1;
	}
}

template <class type> type *allocate(int dimensionCount, Dimension *dimensions) {
	int length = 1;
	for (int i = 0; i < dimensionCount; i++) {
		length *= dimensions[i].length;
	}
	return new type[length];
}

template <class type> type *readArrayFromFile(const char *arrayName,
		int dimensionCount,
		Dimension *dimensions) {

	std::cout << "Enter the file path containing array \"" << arrayName << "\"\n";
	std::cout << "The first line should have its dimension lengths in the form: dim1Length * dime2Length ...\n";
	std::cout << "Subsequent lines should have the data in row major order format\n";
	std::cout << "Elements of array should be separated by spaces\n";

	std::string filePath;
	std::getline(std::cin, filePath);

	std::ifstream file(filePath.c_str());
	if (!file.is_open()) {
		std::cout << "could not open the specified file\n";
		std::exit(EXIT_FAILURE);
	}

	readArrayDimensionInfoFromFile(file, dimensionCount, dimensions);
	type *array = allocate <type> (dimensionCount, dimensions);

	int elementsCount = 1;
	for (int i = 0; i < dimensionCount; i++) {
		elementsCount *= dimensions[i].length;
	}

	int readCount = 0;
	type nextElement;
	while (file >> nextElement) {
		array[readCount] = nextElement;
		readCount++;
	}

	if (readCount < elementsCount) {
		std::cout << "specified file does not have enough data elements: ";
		std::cout << "read only " << readCount << " values\n";
		std::exit(EXIT_FAILURE);
	}

	file.close();
	return array;
}

template <class type> void writeArrayToFile(const char *arrayName,
		type *array,
		int dimensionCount,
		Dimension *dimensions) {

	std::cout << "Enter the file path to write array \"" << arrayName << "\"\n";
	std::string filePath;
	std::getline(std::cin, filePath);

	std::ofstream file(filePath.c_str());
	if (!file.is_open()) {
		std::cout << "could not open the specified file\n";
		std::exit(EXIT_FAILURE);
	}

	int elementsCount = 1;
	int lastDimLength = 0;
	for (int i = 0; i < dimensionCount; i++) {
		if (i > 0) file << " * ";
		lastDimLength = dimensions[i].length;
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
	std::cout << "array \"" << arrayName << "\" has been written on file: " << filePath << '\n';
}


int mainFileReadWrite() {
	Dimension matrixDims[2];
	float *matrix = readArrayFromFile <float> ("matrix", 2, matrixDims);
	for (int i = 0; i < matrixDims[0].length; i++) {
		int rIn = i * matrixDims[1].length;
		for (int j = 0; j < matrixDims[1].length; j++) {
			std::cout << matrix[rIn + j] << '\t';
		}
		std::cout << '\n';
	}
	writeArrayToFile <float> ("matrix", matrix, 2, matrixDims);
	return 0;
}





