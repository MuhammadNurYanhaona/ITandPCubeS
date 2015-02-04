/*
 * utils.h
 *
 *  Created on: Dec 14, 2014
 *      Author: yan
 */

#ifndef UTILS_H_
#define UTILS_H_

#include<deque>

void trim(std::string &str);
bool endsWith(std::string &str, char c);
bool endsWith(std::string &str, std::string &endStr);
void shrinkWhitespaces(std::string &str);
std::deque<std::string> tokenizeString(std::string &str, std::string &delim);
bool startsWith(std::string &str, std::string &endStr);

#endif /* UTILS_H_ */
