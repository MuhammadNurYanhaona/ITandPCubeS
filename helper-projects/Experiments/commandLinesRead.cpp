#include <iostream>
#include <string>
#include <cstdlib>
#include "utils.h"

using namespace std;

int mainCLR(int argc, char *argv[]) {

	for (int i = 1; i < argc; i++) {
		std::string keyValue = string(argv[i]);
		size_t separator = keyValue.find('=');
		if (separator == string::npos) {
			std::cout << "a command line parameter must be in the form of key=value\n";
			std::exit(EXIT_FAILURE);
		}
		std::string key = keyValue.substr(0, separator);
		std::string value = keyValue.substr(separator + 1);
		std::cout << "Key: " << key << "\t\t";
		std::cout << "Value: " << value << "\n";
	}

	return 0;
}
