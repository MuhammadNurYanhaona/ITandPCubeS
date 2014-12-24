#include "code_generator.h"
#include "space_mapping.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void initializeOutputFile(const char *filePath) {
	std::string line;
        std::ifstream commIncludeFile("codegen/default-includes.txt");
	std::ofstream programFile;
        programFile.open (filePath, std::ofstream::out);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "header files included for different purposes" << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	}
	else std::cout << "Unable to open output program file";

	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
			programFile << line << std::endl;
		}
		programFile << std::endl;
	}
	else std::cout << "Unable to open common include file";
	commIncludeFile.close();
	programFile.close();
}

void generateThreadCountMacros(const char *outputFile,       
                MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig) {
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "macro definitions for total and par core thread counts" << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	}
	else std::cout << "Unable to open output program file";
	
	// find lowest PPS to which any LPS has been mapped and highest PPS that has a partitioned LPS mapped
	// to it 
	std::deque<MappingNode*> nodeQueue;
        nodeQueue.push_back(mappingRoot);
	int highestPartitionedPpsId = 1;
	int lowestPpsId = pcubesConfig->Nth(0)->id;
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		PPS_Definition *pps = node->mappingConfig->PPS;
		if (pps->id < lowestPpsId) lowestPpsId = pps->id;
		Space *lps = node->mappingConfig->LPS;
		if (lps->getDimensionCount() > 0 && pps->id > highestPartitionedPpsId) {
			highestPartitionedPpsId = pps->id;
		}
	}
	
	// compute the total number of threads that will participate in computing for the task
	int totalThreads = 1;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->id > highestPartitionedPpsId) continue;
		totalThreads *= pps->units;
		if (pps->id == lowestPpsId) break;
	}
	programFile << "#define Total_Threads " << totalThreads << std::endl;
	
	// determine the number of threads attached par core to understand how to do thread affinity management
	int coreSpaceId = pcubesConfig->Nth(0)->id;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->coreSpace) {
			coreSpaceId = pps->id;
			break;
		}
	}
	int ppsCount = pcubesConfig->NumElements();
	int threadsParCore = 1;
	for (int i = coreSpaceId - 1; i >= lowestPpsId; i--) {
		PPS_Definition *pps = pcubesConfig->Nth(ppsCount - i);
		threadsParCore *= pps->units;
	}	
	programFile << "#define Threads_Par_Core " << threadsParCore << std::endl;
	programFile << std::endl;
	programFile.close();
}

void generateFnForThreadIdsAllocation(const char *outputFile, 
		MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig) {

        std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile;
        
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "function to generate PPU IDs and PPU group IDs for a thread" << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	}
	else std::cout << "Unable to open output program file";

	std::ostringstream functionHeader;
        functionHeader << "ThreadIds *" << "getPpuIdsForThread";
        functionHeader << "(int threadNo)";
        std::ostringstream functionBody;
        
	functionBody << " {\n\n" << statementIndent;
	functionBody << "ThreadIds *threadIds = new ThreadIds";
	functionBody << statementSeparator;
	// allocate a new array to hold the PPU Ids of the thread
	functionBody << statementIndent<< "threadIds->ppuIds = new PPU_ids[Space_Count]" << statementSeparator;
	// declare a local array to hold the index of the thread in different PPS group for ID assignment
	// to be done accurately 
	functionBody << statementIndent << "int idsArray[Space_Count]" << statementSeparator;
	functionBody << statementIndent << "idsArray[Space_Root] = threadNo" << statementSeparator; 

	std::deque<MappingNode*> nodeQueue;
        for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
        	nodeQueue.push_back(mappingRoot->children->Nth(i));
        }

	// declare some local variables needed for thread Id calculation
	functionBody << std::endl;
	functionBody << statementIndent << "int threadCount" << statementSeparator;
	functionBody << statementIndent << "int groupSize" << statementSeparator;
	functionBody << statementIndent << "int groupThreadId" << statementSeparator;
	functionBody << std::endl;

        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }

		PPS_Definition *pps = node->mappingConfig->PPS;
		Space *lps = node->mappingConfig->LPS;
		MappingNode *parent = node->parent;
		Space *parentLps = parent->mappingConfig->LPS;	
		PPS_Definition *parentPps = parent->mappingConfig->PPS;

		// determine the number of partitions current PPS makes to the parent PPS
		int partitionCount = 1;
		int ppsCount = pcubesConfig->NumElements();
		for (int i = parentPps->id - 1; i >= pps->id; i--) {
			partitionCount *= pcubesConfig->Nth(ppsCount - i)->units;
		}

		// create a prefix and variable name to make future references easy
		std::string namePrefix = "threadIds->ppuIds[Space_";
		std::ostringstream varNameStr;
		varNameStr << namePrefix << lps->getName() << "]";
		std::string varName = varNameStr.str();
		std::ostringstream groupThreadIdStr; 
		
		// allocate a variable for PPU-Ids for current PPS
		functionBody << statementIndent << "// for Space " << lps->getName() << statementSeparator;
		functionBody << statementIndent; 
		functionBody << varName << " = new PPU_Ids" << statementSeparator;

		// determine the total number of threads contributing in the parent PPS and current thread's 
		// index in that PPS 
		if (parent == mappingRoot) {
			functionBody << statementIndent << "threadCount = Total_Threads";
			functionBody << statementSeparator;
			groupThreadIdStr << "idsArray[Space_Root]";
		} else {
			functionBody << statementIndent;
			functionBody << "threadCount = " << namePrefix << parentLps->getName() << "]->groupSize";
			functionBody << statementSeparator;
			groupThreadIdStr << "idsArray[Space_" << parentLps->getName() << "]";
		}

		// determine the number of threads per group in the current PPS
		functionBody << statementIndent;
		if (lps->getDimensionCount() > 0) {
			functionBody << "groupSize = threadCount" << " / " << partitionCount;
		} else 	functionBody << "groupSize = threadCount";
		functionBody << statementSeparator;

		// determine the id of the thread in the group it belongs to	
		functionBody << statementIndent;
		functionBody << "groupThreadId = " << groupThreadIdStr.str() << " \% groupSize";
		functionBody << statementSeparator;

		// assign proper group Id and group size in the PPU-Ids variable created before 
		functionBody << statementIndent;
		functionBody  << varName << "->groupId = " << groupThreadIdStr.str() << " / groupSize";
		functionBody << statementSeparator;	
		functionBody << statementIndent;
		functionBody  << varName << "->groupSize = groupSize";
		functionBody << statementSeparator;

		// assign PPU Id to the thread depending on its groupThreadId
		functionBody << statementIndent;
		functionBody << "if (groupThreadId == 0) " << varName << "->id = \n"; 
		functionBody << statementIndent << statementIndent << statementIndent;
		functionBody << varName << "->groupId";
		functionBody << statementSeparator;	
		functionBody << statementIndent;
		functionBody << "else " << varName << "->id = INVALID_ID";
		functionBody << statementSeparator;	
		
		// store the index of the thread in the group for subsequent references	
		functionBody << statementIndent;
		functionBody << "idsArray[Space_" << lps->getName() << "] = groupThreadId";
		functionBody << statementSeparator;
		functionBody << std::endl;
	}
	functionBody << statementIndent << "return threadIds" << statementSeparator;
	functionBody << "}\n";
	
	programFile << std::endl << functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;
	programFile.close();
}
