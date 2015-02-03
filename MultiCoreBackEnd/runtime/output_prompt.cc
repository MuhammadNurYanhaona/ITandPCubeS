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
