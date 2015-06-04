#include "decorator_utils.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string.h>

void decorator::writeSectionHeader(std::ofstream &stream, const char *message) {
	stream << "\n/*----------------------------------------------------------------------------------------------------\n";
        stream << message << '\n';
        stream <<   "----------------------------------------------------------------------------------------------------*/\n";
}

void decorator::writeSubsectionHeader(std::ofstream &stream, const char *message) {
	int messageLength = strlen(message);
	int defaultLength = 100;
	int remaining = 100 - messageLength;
	stream << '\n' << '/' << '/';
	for (int i = 0; i < remaining; i++) stream << "-";
	stream << message;
	stream << '\n';
}
