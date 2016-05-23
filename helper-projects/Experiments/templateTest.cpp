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

void readArrayDimensionInfo(const char *arrayName,
                int dimensionCount, Dimension *dimensions) {

		std::cout << "Enter the dimension lengths for " << dimensionCount << "D array \"" << arrayName << "\"\n";
        std::cout << "The format is \"";
        for (int i = 0; i < dimensionCount; i++) {
                if (i > 0) std::cout << " * ";
                std::cout << "Dim" << i + 1 << "-Range";
        }
        std::cout << "\"\n";
        std::cout << "Here a range has the form \"Start-Index:End-Index\"\n";
        std::string input;
        std::getline(std::cin, input);
        std::string delim1 = "*";
        std::string delim2 = ":";

        std::deque<std::string> dimensionList = tokenizeString(input, delim1);
        for (int i = 0; i < dimensionCount; i++) {
        	std::string rangeToken = dimensionList.front();
        	std::deque<std::string> rangeBoundaries = tokenizeString(rangeToken, delim2);
        	std::string min = rangeBoundaries.front();
        	trim(min);
        	dimensions[i].range.min = atoi(min.c_str());
        	rangeBoundaries.pop_front();
        	std::string max = rangeBoundaries.front();
        	trim(max);
        	dimensions[i].range.max = atoi(max.c_str());
        	dimensionList.pop_front();
        	dimensions[i].length = abs(dimensions[i].range.max - dimensions[i].range.min) + 1;
        }
}

bool readBoolean(const char* varName) {
	std::cout << "Enter value for boolean variable " << varName << " as True or False\n";
	std::string value;
	std::getline(std::cin, value);
	trim(value);
	return strcmp(value.c_str(), "True") == 0;
}

int mainTemplate() {
	int quantity;
//	float price;
	bool flag = readBoolean("Flag");
//	quantity = readPrimitive<int>("Quantity");
//	price = readPrimitive<float>("Price");
//	std::cout << "Quantity is " << quantity << " and price is " << price << "\n";
	std::cout << "Flag is " << flag << "\n";

//	Dimension matrix[2];
//	readArrayDimensionInfo("matrix", 2, matrix);
//	std::cout << "Dimension 1: from " << matrix[0].range.min << " to " << matrix[0].range.max;
//	std::cout << " length " << matrix[0].length;
//	std::cout << "\nDimension 2: from " << matrix[1].range.min << " to " << matrix[1].range.max;
//		std::cout << " length " << matrix[1].length;
 	return 0;
}

template <class type> type readPrimitive(const char *varName) {
	std::cout << "Enter value for " << varName << "\n";
	type value;
	std::cin >> value;
	return value;
}

