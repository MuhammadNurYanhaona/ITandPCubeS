#include "input_prompt.h"
#include "output_prompt.h"
#include "structure.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

using namespace inprompt;

void inprompt::readArrayDimensionInfo(const char *arrayName, int dimensionCount, Dimension *dimensions) {

	std::cout << "Enter the dimension lengths for " << dimensionCount << "D array \"" << arrayName << "\"\n";
        std::cout << "The format is \"";
        for (int i = 0; i < dimensionCount; i++) {
                if (i > 0) std::cout << " * ";
                std::cout << "Dim" << i + 1 << "-Range";
        }

        std::cout << "\"\n";
        std::cout << "Here a range has the form \"Start-Index:End-Index\"\n";
        std::string input;
	outprompt::readNonEmptyLine(input);
        std::string delim1 = "*";
        std::string delim2 = ":";

        List<std::string> *dimensionList = string_utils::tokenizeString(input, delim1);
        for (int i = 0; i < dimensionCount; i++) {
        	std::string rangeToken = dimensionList->Nth(i);
        	List<std::string> *rangeBoundaries = string_utils::tokenizeString(rangeToken, delim2);
        	std::string min = rangeBoundaries->Nth(0);
        	string_utils::trim(min);
        	dimensions[i].range.min = atoi(min.c_str());
        	std::string max = rangeBoundaries->Nth(1);
        	string_utils::trim(max);
        	dimensions[i].range.max = atoi(max.c_str());
		dimensions[i].setLength();
        }
}

bool inprompt::readBoolean(const char *varName) {
	std::cout << "Enter value for boolean variable \"" << varName << "\" as True or False\n";
	std::string value;
	outprompt::readNonEmptyLine(value);
	string_utils::trim(value);
	return strcmp(string_utils::toLower(value.c_str()), "true") == 0;
}

const char *inprompt::readString(const char *varName) {
	std::cout << "Enter value for \"" << varName << '"' << std::endl;
	std::string value;
	outprompt::readNonEmptyLine(value);
	string_utils::trim(value);
	return strdup(value.c_str());
}

void inprompt::readArrayDimensionInfoFromFile(std::ifstream &file, int dimensionCount, Dimension *dimensions) {
        std::string input;
        std::getline(file, input);
        std::string delim = "*";
        List<std::string> *dimensionList = string_utils::tokenizeString(input, delim);
        for (int i = 0; i < dimensionCount; i++) {
                std::string token = dimensionList->Nth(i);
                dimensions[i].range.min = 0;
                string_utils::trim(token);
                dimensions[i].range.max = atoi(token.c_str()) - 1;
		dimensions[i].setLength();
        }
}

