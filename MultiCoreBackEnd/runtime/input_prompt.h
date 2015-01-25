#ifndef _H_input_prompt
#define _H_input_prompt

/* This header compiles library functions to read array metadata, scalar variable values, lists, 
   and user defined objects from external files and console  
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


namespace inprompt {
	
	// read and set the dimension lengths of some array having a given number of dimensions
	void readArrayDimensionInfo(const char *arrayName, int dimensionCount, Dimension *dimensions);
	
	// read and return a primitive type from the console
	template <class type> type readPrimitive(const char *varName) {
        	std::cout << "Enter value for " << varName << "\n";
        	type value;
        	std::cin >> value;
        	return value;
	}

	// read data for an array from an input file; TODO not implemented yet
	template <class type> void readArray(const char *arrayName, 
			int dimensionCount, Dimension *dimensions, type *array);

	// read a boolean value from the console and return it
	bool readBoolean(const char *varName);
}

#endif
