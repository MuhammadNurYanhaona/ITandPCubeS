#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>
#include "utils.h"

using namespace std;

int mainPFR() {

	string propertyLine;
	std::ifstream propertiesFile("/home/yan/deployment.properties");
	std::string delimiter = "=";

	if (!propertiesFile.is_open()) {
		cout << "Unable to open properties file\n";
		exit(EXIT_FAILURE);
	} else {
		while ( getline (propertiesFile,propertyLine) ) {
			std::deque<std::string> tokenList = tokenizeString(propertyLine, delimiter);
			string key = tokenList.front();
			trim(key);
			tokenList.pop_front();
			string value = tokenList.front();
			trim(value);
			cout << "Key: " << key << "\tValue: " << value << '\n';
		}
		propertiesFile.close();
	}

	return 0;
}
