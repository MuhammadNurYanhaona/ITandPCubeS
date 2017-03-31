#include "decorator_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string.h>

void decorator::writeSectionHeader(std::ofstream &stream, const char *message) {
	stream << "\n/*--------------------------------------------------------------------------------------------------------------\n";
        stream << message << '\n';
        stream <<   "--------------------------------------------------------------------------------------------------------------*/\n";
}

void decorator::writeSectionHeader(std::ostringstream &stream, const char *message) {
	stream << "\n/*--------------------------------------------------------------------------------------------------------------\n";
        stream << message << '\n';
        stream <<   "--------------------------------------------------------------------------------------------------------------*/\n";
}

void decorator::writeSubsectionHeader(std::ofstream &stream, const char *message) {
	int messageLength = strlen(message);
	int defaultLength = 110;
	int remaining = defaultLength - messageLength;
	stream << '\n' << '/' << '/';
	for (int i = 0; i < remaining; i++) stream << "-";
	stream << message;
	stream << '\n';
}

void decorator::writeCommentHeader(int indentLevel, std::ostream *stream, const char *message) {
	int messageLength = strlen(message);
	int defaultLength = 110;
	int tabSpace = indentLevel * 8;
	int remaining =  defaultLength - messageLength - tabSpace;
	*stream << '\n';
	for (int i = 0; i < indentLevel; i++) *stream << '\t';
	*stream << '/' << '/';
	for (int i = 0; i < remaining; i++) *stream << "-";
	*stream << message;
	*stream << '\n';
}

void decorator::writeCommentHeader(std::ostream *stream, const char *message, const char *indentStr) {
	int messageLength = strlen(message);
	int defaultLength = 110;
	int tabSpace = strlen(indentStr) * 8;
	int remaining =  defaultLength - messageLength - tabSpace;
	*stream << '\n' << indentStr;
	*stream << '/' << '/';
	for (int i = 0; i < remaining; i++) *stream << "-";
	*stream << message;
	*stream << '\n';
}
