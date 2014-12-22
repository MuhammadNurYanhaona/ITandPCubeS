#include "string_utils.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>

void string_utils::trim(std::string &str) {

	// remove all white spaces at the end
	while (true) {
		if (std::isspace(str[0])) {
			str.erase(0, 1);
		} else break;
	}
	// remove all white spaces at the end
	while (true) {
		if (std::isspace(str[str.length() - 1])) {
			str.erase(str.length() - 1, 1);
		} else break;
	}
}

bool string_utils::endsWith(std::string &str, char c) {
        return str[str.length() - 1] == c;
}

bool string_utils::endsWith(std::string &str, std::string &endStr) {
        if (str.length() < endStr.length()) return false;
        int strLength = str.length();
        int endStrLength = endStr.length();
        for (int i = 0; i < endStrLength; i++) {
        		char c = endStr[endStrLength - i - 1];
                if (str[strLength - i - 1] != c) return false;
        }
        return true;
}

bool string_utils::startsWith(std::string &str, std::string &endStr) {
	if (str.length() < endStr.length()) return false;
        int endStrLength = endStr.length();
        for (int i = 0; i < endStrLength; i++) {
                if (str[i] != endStr[i]) return false;
        }
        return true;
}

void string_utils::shrinkWhitespaces(std::string &str) {
	for (int i = 0; i < str.length(); i++) {
		if (!std::isspace(str[i])) continue;
		str.replace(i, 1, " ");
		int j = i + 1;
		while (j < str.length()) {
			if (std::isspace(str[j])) {
				str.erase(j, 1);
			} else break;
		}
	}
}

List<std::string> *string_utils::tokenizeString(std::string &str, std::string &delim) {
	List<std::string> *tokenList = new List<std::string>;
	std::string strCp;
	strCp.assign(str);
	size_t pos = 0;
	std::string token;
	while ((pos = strCp.find(delim)) != std::string::npos) {
		token = strCp.substr(0, pos);
		trim(token);
		shrinkWhitespaces(token);
		if (token.length() > 0) tokenList->Append(token);
		strCp.erase(0, pos + delim.length());
	}
	trim(strCp);
	if (strCp.length() > 0) tokenList->Append(strCp);
	return tokenList;
}
