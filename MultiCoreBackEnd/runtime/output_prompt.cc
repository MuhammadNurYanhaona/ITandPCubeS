#include "output_prompt.h"
#include "../codegen/structure.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

using namespace outprompt;

void outprompt::readNonEmptyLine(std::string &line) {
	while (true) {
		std::getline(std::cin, line);
		string_utils::trim(line);
		if (line.length() > 0) break;
	}
}

bool outprompt::getYesNoAnswer(const char *prompt) {
	std::cout << prompt << std::endl;
	std::cout << "Type 'Y' for yes or 'N' for no\n";
	std::string response;
	readNonEmptyLine(response);
	string_utils::trim(response);
	if ('Y' == response[0]) return true;
	return false;
}
