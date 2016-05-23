#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include "utils.h"
#include <string.h>

int tabSpace = 8;
int idealLineLength = 60;
int maximumLineLength = 80;

const char* breakLongLine(int indent, std::string originalLine) {

	std::string line(originalLine);
	int length = line.length() + indent * tabSpace - indent;
	if (length <= maximumLineLength) return line.c_str();
	int brokenLineLength = idealLineLength - indent * tabSpace;
	std::ostringstream stream;
	std::string delim = " ";
	size_t pos;
	std::string token;
	bool first = true;
	while ((pos = line.find(delim, brokenLineLength)) != std::string::npos) {
		token = line.substr(0, pos);
		trim(token);
		line.erase(0, pos);
		if (first) {
			for (int i = 0; i < indent; i++) stream << "\t";
			indent += 2;
			brokenLineLength = idealLineLength - indent * tabSpace;
			first = false;
		} else {
			stream << "\n";
			for (int i = 0; i < indent; i++) stream << "\t";
		}
		stream << token;
	}
	trim(line);
	if (line.length() > 0) {
		stream << "\n";
		for (int i = 0; i < indent; i++) stream << "\t";
		stream << line;
	}

	return strdup(stream.str().c_str());
}

int mainBL() {
	std::string line = "There are times when you feel like things are not going so well here so what to do about it can you give me a solution?";
	std::cout << breakLongLine(2, line);
	return 0;
}

