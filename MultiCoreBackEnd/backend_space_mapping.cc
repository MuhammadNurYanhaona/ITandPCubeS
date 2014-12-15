#include "backend_space_mapping.h"
#include "backend_structure.h"
#include "task_space.h"
#include "list.h"
#include "string.h"
#include "string_utils.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

List<PPS_Definition*> *parsePCubeSDescription(const char *filePath) {

	List<PPS_Definition*> *list = new List<PPS_Definition*>;
	std::string line;
	std::ifstream pcubesfile(filePath);
	std::string separator1 = ":";
	std::string separator2 = " ";
	std::string separator3 = "(";
	List<std::string> *tokenList;

	if (!pcubesfile.is_open()) std::cout << "could not open PCubeS specification file" << std::endl;

	while (std::getline(pcubesfile, line)) {
		// trim line and escape it if it is a comment	
		string_utils::trim(line);
		if (line.length() == 0) continue;
		std::string comments = "//";
		if (string_utils::startsWith(line, comments)) continue;

		// separate space number from its name and PPU count
		tokenList = string_utils::tokenizeString(line, separator1);
		std::string spaceNoStr = tokenList->Nth(0);
		std::string spaceNameStr = tokenList->Nth(1);

		// retrieve the space ID; also determine if current space represents CPU cores
		tokenList = string_utils::tokenizeString(spaceNoStr, separator2);
		std::string spaceNo = tokenList->Nth(1);
		int spaceId;
		bool coreSpace = false;
		if (string_utils::endsWith(spaceNo, '*')) {
			spaceNo = spaceNo.substr(0, spaceNo.length() - 1);
			spaceId = atoi(spaceNo.c_str());
			coreSpace = true;
		} else {
			spaceId = atoi(spaceNo.c_str());
		}

		// retrieve space name and PPU count
		tokenList = string_utils::tokenizeString(spaceNameStr, separator3);
		std::string spaceName = tokenList->Nth(0);
		std::string ppuCountStr = tokenList->Nth(1);
		int countEnd = ppuCountStr.find(')');
		int ppuCount = atoi(ppuCountStr.substr(0, countEnd).c_str());

		// create a PPS definition
		PPS_Definition *spaceDefinition = new PPS_Definition();
		spaceDefinition->id = spaceId;
		spaceDefinition->name = strdup(spaceName.c_str());
		spaceDefinition->units = ppuCount;
		spaceDefinition->coreSpace = coreSpace;
			
		// store the space definition in the list in top-down order
		int i = 0;	
		for (; i < list->NumElements(); i++) {
			if (list->Nth(i)->id > spaceId) continue;
			else break;	
		}
		list->InsertAt(spaceDefinition, i);
	}
	pcubesfile.close();

	for (int i = 0; i < list->NumElements(); i++) {
		printf("Space %s-%d-%d\n", list->Nth(i)->name, list->Nth(i)->id, list->Nth(i)->units);
	}
	return list;
}
