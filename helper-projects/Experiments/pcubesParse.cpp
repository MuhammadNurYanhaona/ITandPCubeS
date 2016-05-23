#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "string.h"

#include "structures.h"
#include "utils.h"

void parsePCubeSDescription(const char *filePath) {

	std::string line;
	std::ifstream pcubesfile(filePath);
	std::string separator1 = ":";
	std::string separator2 = " ";
	std::string separator3 = "(";
	std::deque<std::string> tokenList;

	if (pcubesfile.is_open()) {
		while (std::getline(pcubesfile, line)) {

			trim(line);
			if (line.length() == 0) continue;
			std::string comments = "//";
			if (startsWith(line, comments)) continue;

			// separate space number from its name and PPU count
			tokenList = tokenizeString(line, separator1);
			std::string spaceNoStr = tokenList.front();
			tokenList.pop_front();
			std::string spaceNameStr = tokenList.front();

			// retrieve the space ID; also set the global variable if the
			// current space represent CPU cores
			tokenList = tokenizeString(spaceNoStr, separator2);
			tokenList.pop_front();
			std::string spaceNo = tokenList.front();
			int spaceId;
			if (endsWith(spaceNo, '*')) {
				spaceNo = spaceNo.substr(0, spaceNo.length() - 1);
				spaceId = atoi(spaceNo.c_str());
				std::cout << "Core Space ";
			} else {
				spaceId = atoi(spaceNo.c_str());
			}

			// retrieve space name and PPU count
			tokenList = tokenizeString(spaceNameStr, separator3);
			std::string spaceName = tokenList.front();
			tokenList.pop_front();
			std::string ppuCountStr = tokenList.front();
			int countEnd = ppuCountStr.find(')');
			int ppuCount = atoi(ppuCountStr.substr(0, countEnd).c_str());

			PPS_Definition *spaceDefinition = new PPS_Definition();
			spaceDefinition->id = spaceId;
			spaceDefinition->name = strdup(spaceName.c_str());
			spaceDefinition->units = ppuCount;

			std::cout << spaceId << "-" << spaceName << "-" << ppuCount << std::endl;
		}

		pcubesfile.close();
	}
	else std::cout << "Unable to open file";
}

int main8() {
	parsePCubeSDescription("/home/yan/pcubes.ml");
	return 0;
}
