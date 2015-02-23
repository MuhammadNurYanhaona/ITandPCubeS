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
#include "fileUtility.h"

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



