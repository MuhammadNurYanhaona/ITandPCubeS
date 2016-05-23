#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include "utils.h"


int main4() {
	std::string line;
	std::ifstream myfile("/home/yan/pcubes.ml");
	std::string delimiter = ":";
	std::string token;

	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			size_t pos = line.find(":");
			std::string spacePart = line.substr(0, pos);

			std::cout << spacePart << " || ";
			line.erase(0, pos + delimiter.length());
			std::cout << line << std::endl;
		}
		myfile.close();
	} else {
		std::cout << "Unable to open file";
		exit(-1);
	}

	std::string str = "  	 there is goodness			   ";
	trim(str);
	std::cout << str;
	str = "habai endds";
	std::string str2 = "ends";
	if (endsWith(str, str2)) {
		std::cout << "Yes it does ends";
	}
	return 0;
}

void trim(std::string &str) {
	// remove all white spaces at the end
	while (true) {
		if (std::isspace(str[0])) {
			str.erase(0, 1);
		} else break;
	}
	// remove all white spaces at the end
	while (true) {
		if (isspace(str[str.length() - 1])) {
			str.erase(str.length() - 1, 1);
		} else break;
	}
}

bool endsWith(std::string &str, char c) {
        return str[str.length() - 1] == c;
}

bool endsWith(std::string &str, std::string &endStr) {
        if (str.length() < endStr.length()) return false;
        int strLength = str.length();
        int endStrLength = endStr.length();
        for (int i = 0; i < endStrLength; i++) {
        		char c = endStr[endStrLength - i - 1];
                if (str[strLength - i - 1] != c) return false;
        }
        return true;
}

bool startsWith(std::string &str, std::string &endStr) {
        if (str.length() < endStr.length()) return false;
        int strLength = str.length();
        int endStrLength = endStr.length();
        for (int i = 0; i < endStrLength; i++) {
                if (str[i] != endStr[i]) return false;
        }
        return true;
}
