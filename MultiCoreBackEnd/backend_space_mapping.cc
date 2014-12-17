#include "backend_space_mapping.h"
#include "backend_structure.h"
#include "task_space.h"
#include "list.h"
#include "string.h"
#include "string_utils.h"
#include "hashtable.h"

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

MappingNode *parseMappingConfiguration(const char *taskName,
                const char *filePath,
                PartitionHierarchy *lpsHierarchy,
                List<PPS_Definition*> *pcubesConfig) {

	std::string line;
	std::ifstream mappingfile(filePath);
	std::string commentsDelimiter = "//";
	std::string newlineDelimiter = "\n";
	std::string mappingDelimiter = ":";
	List<std::string> *mappingList;
	List<std::string> *tokenList;
	std::string description;

	// open the mapping configuration file and read mapping configurations in a string
	if (mappingfile.is_open()) {
		while (std::getline(mappingfile, line)) {
			string_utils::trim(line);
			if (line.length() == 0) continue;
			if (string_utils::startsWith(line, commentsDelimiter)) continue;
			mappingList = string_utils::tokenizeString(line, commentsDelimiter);
			description.append(mappingList->Nth(0));
			description.append("\n");
		}
	} else {
		std::cout << "could not open the mapping file.\n";
	}

	// locate the mapping configuration of the mentioned task and extract it
	int taskConfigBegin = description.find(taskName);
	int mappingStart = description.find('{', taskConfigBegin);
	int mappingEnd = description.find('}', taskConfigBegin);
	std::string mapping = description.substr(mappingStart + 1, mappingEnd - mappingStart - 1);
	string_utils::trim(mapping);

	// create the root of the mapping hierarchy
	MapEntry *rootEntry = new MapEntry();
	Space *rootSpace = lpsHierarchy->getRootSpace();
	rootEntry->LPS = rootSpace;
	rootEntry->PPS = pcubesConfig->Nth(0);
	MappingNode *rootNode = new MappingNode();
	rootNode->parent = NULL;
	rootNode->mappingConfig = rootEntry;
	rootNode->children = new List<MappingNode*>;

	// parse individual lines and construct the mapping hierarchy
	mappingList = string_utils::tokenizeString(mapping, newlineDelimiter);
	Hashtable<MappingNode*> *mappingTable = new Hashtable<MappingNode*>;
	int i = 0;
	int mappingCount = mappingList->NumElements();
	int totalPPSes = pcubesConfig->NumElements();
	while (i < mappingCount) {
		// determine the LPS and PPS for the mapping
		std::cout << mappingList->Nth(i) << std::endl;
		std::string mapping = mappingList->Nth(i);
		tokenList = string_utils::tokenizeString(mapping, mappingDelimiter);
		std::string lpsStr = tokenList->Nth(0);
		char lpsId = lpsStr.at(lpsStr.length() - 1);
		Space *lps = lpsHierarchy->getSpace(lpsId);
		int ppsId = atoi(tokenList->Nth(1).c_str());
		PPS_Definition *pps = pcubesConfig->Nth(totalPPSes - ppsId);
		
		// create a mapping configuration object
		MapEntry *entry = new MapEntry();
		entry->LPS = lps;
		entry->PPS = pps;
		MappingNode *node = new MappingNode();
		node->parent = NULL;
		node->mappingConfig = entry;
		node->children = new List<MappingNode*>;
		mappingTable->Enter(lps->getName(), node, true);

		std::cout << ppsId << "--" << lpsId;
		std::cout << "-----------------\n";
		i++;
	}

	// correct the mapping hierarchy by setting parent and children references correctly
	MappingNode *currentNode = NULL;
	Iterator<MappingNode*> iterator = mappingTable->GetIterator();
	while ((currentNode = iterator.GetNextValue()) != NULL) {
		Space *parentLps = currentNode->mappingConfig->LPS->getParent();
		if (rootSpace == parentLps) {
			currentNode->parent = rootNode;
			rootNode->children->Append(currentNode);
		} else {
			MappingNode *parent = mappingTable->Lookup(parentLps->getName());
			currentNode->parent = parent;
			parent->children->Append(currentNode);
		}
	} 

	return rootNode;
}
