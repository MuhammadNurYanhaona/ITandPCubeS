#include <sstream>
#include <iostream>
#include <string>
#include <deque>
#include "utils.h"

std::deque<std::string> tokenizeString(std::string &str, std::string &delim) {
	std::deque<std::string> tokenList;
	std::string strCp;
	strCp.assign(str);
	size_t pos = 0;
	std::string token;
	while ((pos = strCp.find(delim)) != std::string::npos) {
		token = strCp.substr(0, pos);
		trim(token);
		shrinkWhitespaces(token);
		if (token.length() > 0) tokenList.push_back(token);
		strCp.erase(0, pos + delim.length());
	}
	trim(strCp);
	if (strCp.length() > 0) tokenList.push_back(strCp);
	return tokenList;
}

int main6 () {
	std::string s = " scott   pilgrim >=	  tiger		>=mushroom   >=   ";
	std::string delimiter = ">=";
	std::deque<std::string> tokenList = tokenizeString(s, delimiter);
	while (!tokenList.empty()) {
		std::cout << tokenList.front() << std::endl;
		tokenList.pop_front();
	}
}


