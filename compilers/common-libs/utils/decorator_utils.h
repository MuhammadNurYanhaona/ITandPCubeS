#ifndef _H_decorator_utils
#define _H_decorator_utils

#include <iostream>
#include <fstream>
#include <sstream>

namespace decorator {

	void writeSectionHeader(std::ofstream &stream, const char *message);
	void writeSectionHeader(std::ostringstream &stream, const char *message);
	
	void writeSubsectionHeader(std::ofstream &stream, const char *message);

	void writeCommentHeader(int indentLevel, std::ostream *stream, const char *message);
	void writeCommentHeader(std::ostream *stream, const char *message, const char *indentStr);
}

#endif
