#ifndef _H_string_utils
#define _H_string_utils

#include <iostream>
#include "list.h"

/* A library of string functions that are useful for the compiler
*/

namespace string_utils {

	const int TAB_SPACE = 8;
	const int IDEAL_LINE_LENGTH = 60;
	const int MAXIMUM_LINE_LENGTH = 80;
	
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

	// break a long line into multiple lines without compromising the content
	const char *breakLongLine(int indent, std::string originalLine);

	// generate a list of attributes from a text containing each attribute 
	// in <attribute> format
	List<const char*> *readAttributes(std::string &str);

	// determine if a list of string contains a particular string
	bool contains(List<const char*> *list, const char *str);

	// compute the union of two lists and assign the result to the first list
	void combineLists(List<const char*> *list1, List<const char*> *list2);
}

#endif
