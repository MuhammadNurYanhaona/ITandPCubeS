#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "string.h"

#include "structures.h"
#include "utils.h"

void parseMappingConfiguration(const char *filePath, const char *taskName) {

	std::string line;
	std::ifstream mappingfile(filePath);
	std::string commentsDelimiter = "//";
	std::string newlineDelimiter = "\n";
	std::string mappingDelimiter = ":";
	std::deque<std::string> mappingList;
	std::deque<std::string> tokenList;
	std::string description;

	// open the mapping configuration file and read mapping configurations in a string
	if (mappingfile.is_open()) {
		while (std::getline(mappingfile, line)) {
			trim(line);
			if (line.length() == 0) continue;
			if (startsWith(line, commentsDelimiter)) continue;
			mappingList = tokenizeString(line, commentsDelimiter);
			description.append(mappingList.front());
			description.append("\n");
		}
	} else {
		std::cout << "could not open the mapping file.\n";
	}

	// locate the mapping configuration of the mentioned task and extract it
	int taskConfigBegin = description.find(taskName);
	int mappingStart = description.find('{', taskConfigBegin);
	int mappingEnd = description.find('}', taskConfigBegin);
	std::string mapping = description.substr(mappingStart + 1, mappingEnd - mappingStart - 1);
	trim(mapping);

	// parse individual lines and construct the mapping hierarchy
	mappingList = tokenizeString(mapping, newlineDelimiter);
	while (!mappingList.empty()) {
		std::cout << mappingList.front() << std::endl;
		tokenList = tokenizeString(mappingList.front(), mappingDelimiter);
		std::string lpsStr = tokenList.front();
		tokenList.pop_front();
		char lps = lpsStr.at(lpsStr.length() - 1);
		int pps = atoi(tokenList.front().c_str());
		std::cout << pps << "--" << lps;
		std::cout << "-----------------\n";
		mappingList.pop_front();
	}
}

int mainMP() {
	parseMappingConfiguration("/home/yan/opteron-solver-mapping.map", "LU Factorization");
	return 0;
}
