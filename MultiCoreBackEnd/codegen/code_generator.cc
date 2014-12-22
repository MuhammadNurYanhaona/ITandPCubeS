#include "code_generator.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void initializeOutputFile(const char *filePath) {
	std::string line;
        std::ifstream commIncludeFile("codegen/default-includes.txt");
	std::ofstream programFile;
        programFile.open (filePath, std::ofstream::out);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "header files included for different purposes" << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	}
	else std::cout << "Unable to open output program file";

	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
			programFile << line << std::endl;
		}
		programFile << std::endl;
	}
	else std::cout << "Unable to open common include file";
	commIncludeFile.close();
	programFile.close();
}
