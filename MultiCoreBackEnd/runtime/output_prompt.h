#ifndef _H_output_prompt
#define _H_output_prompt

// this header compiles all routines that prompt a user to enter output file locations for his program
// output and write results on those files.

#include "../codegen/structure.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

namespace outprompt {

	// read a nonempty line from the console
        void readNonEmptyLine(std::string &line);

	// write a prompt to determine if the user want to do something or not
	bool getYesNoAnswer(const char *prompt);

	// this function, as its name suggests, write an array of arbitrary dimensions, containing arbitrary
	// primitive type contents on a file
	template <class type> void writeArrayToFile(const char *arrayName,
			type *array,
			int dimensionCount,
			Dimension *dimensions) {

		std::cout << "Enter the file path to write array \"" << arrayName << "\"\n";
		std::string filePath;
		readNonEmptyLine(filePath);

		std::ofstream file(filePath.c_str());
		if (!file.is_open()) {
			std::cout << "could not open output file: \"" << filePath << "\"\n";
			std::exit(EXIT_FAILURE);
		}

		int elementsCount = 1;
		int lastDimLength = 0;
		for (int i = 0; i < dimensionCount; i++) {
			if (i > 0) file << " * ";
			lastDimLength = dimensions[i].getLength();
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

}

#endif
