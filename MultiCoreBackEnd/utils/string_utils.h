#ifndef _H_string_utils
#define _H_string_utils

#include <iostream>
#include "list.h"

/* A library of string functions that are useful for the compiler
*/

namespace string_utils {
	
	// remove whitespaces from the beginning and the end of any string
	void trim(std::string &str);

	// check if a string ends with any particular character or string
	bool endsWith(std::string &str, char c);
	bool endsWith(std::string &str, std::string &endStr);

	// check if a string starts with any particular string
	bool startsWith(std::string &str, std::string &endStr);

	// replace more than one whitespaces in a row with a single space in
	// any place within a string
	void shrinkWhitespaces(std::string &str);

	// tokenize a string, this implementation trim and remove extra 
	// whitespaces within the tokens.	
	List<std::string> *tokenizeString(std::string &str, std::string &delims);

	// replace all occurances of a character with another character
	const char *replaceChar(const char *origString, char ch1, char ch2);

	// convert a character string to lower case and return a new string
	const char *toLower(const char *origString);

	// get the capitalized characters of string as its initials
	const char *getInitials(const char *str);

	int getLastIndexOf(const char *str, char ch);

	// get the substring from begin to end with both indexes included
	char *substr(const char *str, int begin, int end);
}

#endif
