#ifndef _H_input_prompt
#define _H_input_prompt

/* This header compiles library functions to read array metadata, scalar variable values, lists, and
   user defined objects from external files and console  
*/

#include "../codegen/structure.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

// Note that when templated functions are put within a header file, their definitions should accompany
// their declarations. Otherwise, the C++ compiler we are using cannot resolve the template types.

namespace inprompt {

	// read and set the dimension lengths of some array having a given number of dimensions
	void readArrayDimensionInfo(const char *arrayName, int dimensionCount, Dimension *dimensions);
	
	// read and return a primitive type from the console
	template <class type> type readPrimitive(const char *varName) {
        	std::cout << "Enter value for \"" << varName << "\"\n";
        	type value;
        	std::cin >> value;
        	return value;
	}

	// read a boolean value from the console and return it
	bool readBoolean(const char *varName);

	// read a string from the console and return it
	const char *readString(const char *varName);

	// read dimension length information from the first line of an already openned file and construct
	// default dimension ranges (increasing and starting from 0) using length information 
	void readArrayDimensionInfoFromFile(std::ifstream &file, int dimensionCount, Dimension *dimensions);

	// read an array of arbitrary dimensions from a file; the dimension reference variable must be 
	// passed along to be properly initialized 
	template <class type> type *readArrayFromFile(const char *arrayName,
			int dimensionCount,
			Dimension *dimensions, const char *fileName = NULL) {

		std::string filePath;
		if (fileName == NULL) {
			std::cout << "Enter the file path containing array \"" << arrayName << "\"\n";
			std::cout << "The first line should have its dimension lengths in the form: ";
			std::cout << "dim1Length * dime2Length ...\n";
			std::cout << "Subsequent lines should have the data in row major order format\n";
			std::cout << "Elements of array should be separated by spaces\n";
			std::getline(std::cin, filePath);
		} else {
			filePath = std::string(fileName);
		}

		std::ifstream file(filePath.c_str());
		if (!file.is_open()) {
			std::cout << "could not open the specified file\n";
			std::exit(EXIT_FAILURE);
		}

		readArrayDimensionInfoFromFile(file, dimensionCount, dimensions);
		int elementsCount = 1;
		for (int i = 0; i < dimensionCount; i++) {
			elementsCount *= dimensions[i].getLength();
		}
		type *array = new type[elementsCount];

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
}

#endif
