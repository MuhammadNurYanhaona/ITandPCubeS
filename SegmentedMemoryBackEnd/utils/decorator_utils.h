#ifndef _H_decorator_utils
#define _H_decorator_utils

#include <fstream>
#include <sstream>

namespace decorator {

	void writeSectionHeader(std::ofstream &stream, const char *message);
	void writeSubsectionHeader(std::ofstream &stream, const char *message);
}

#endif
