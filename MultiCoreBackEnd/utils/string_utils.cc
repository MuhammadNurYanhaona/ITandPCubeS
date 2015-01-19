#include "string_utils.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <string.h>
#include <stdio.h>

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

const char *string_utils::replaceChar(const char *origString, char ch1, char ch2) {
	std::string str(origString);
	for (int i = 0; i < str.length(); i++) {
		if (str[i] != ch1) continue;
		str[i] = ch2;
	}
	return strdup(str.c_str());
}

const char *string_utils::toLower(const char *origString) {
	std::string str(origString);
	for (int i = 0; i < str.length(); i++) {
		char ch = str[i];
		char lowerCh = tolower(ch);
		str[i] = lowerCh;	
	}
	return strdup(str.c_str());
}

const char *string_utils::getInitials(const char *str) {
	std::ostringstream initials;
	int length = strlen(str);
	for (int i = 0; i < length; i++) {
		char ch = str[i];
		if (ch >= 'A' && ch <= 'Z') {
			initials << ch;
		}
	}
	return strdup(initials.str().c_str());
}

int string_utils::getLastIndexOf(const char *str, char ch) {
	int length = strlen(str);
	int lastIndex = -1;
	for (int i = 0; i < length; i++) {
		if (str[i] == ch) lastIndex = i;
	}
	return lastIndex;
}

char *string_utils::substr(const char *str, int begin, int end) {
	int length = end - begin + 1;
	char *buffer = new char[length + 1];
	const char *source = str + begin;
	strncpy(buffer, source, length);
	return buffer;
}

const char* string_utils::breakLongLine(int indent, std::string originalLine) {

	std::string line(originalLine);
	int length = line.length() + indent * TAB_SPACE - indent;
	if (length <= MAXIMUM_LINE_LENGTH) return line.c_str();
	int brokenLineLength = IDEAL_LINE_LENGTH - indent * TAB_SPACE;
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
			brokenLineLength = IDEAL_LINE_LENGTH - indent * TAB_SPACE;
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
