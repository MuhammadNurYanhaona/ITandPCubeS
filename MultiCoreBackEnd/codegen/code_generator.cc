#include "code_generator.h"
#include "space_mapping.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"

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
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}

	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
			programFile << line << std::endl;
		}
		programFile << std::endl;
	} else {
		std::cout << "Unable to open common include file";
		std::exit(EXIT_FAILURE);
	}
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
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
	
	// find lowest PPS to which any LPS has been mapped and highest PPS that has an un-partitioned LPS 
	// mapped to it 
	std::deque<MappingNode*> nodeQueue;
        nodeQueue.push_back(mappingRoot);
	int lowestPpsId = pcubesConfig->Nth(0)->id;
	int highestPartitionedPpsId = 1;
	int highestUnpartitionedPpsId = lowestPpsId; // the top-most PPS handling the root LPS
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
		} else if (lps->getDimensionCount() == 0) {
			if (pps->id > highestPartitionedPpsId && pps->id < highestUnpartitionedPpsId) {
				highestUnpartitionedPpsId = pps->id;
			}
		}
	}
	
	// compute the total number of threads that will participate in computing for the task
	int totalThreads = 1;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->id >= highestUnpartitionedPpsId) continue;
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
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}

	std::ostringstream functionHeader;
        functionHeader << "ThreadIds *" << "getPpuIdsForThread";
        functionHeader << "(int threadNo)";
        std::ostringstream functionBody;
        
	functionBody << " {\n\n" << statementIndent;
	functionBody << "ThreadIds *threadIds = new ThreadIds";
	functionBody << statementSeparator;
	// allocate a new array to hold the PPU Ids of the thread
	functionBody << statementIndent<< "threadIds->ppuIds = new PPU_Ids[Space_Count]" << statementSeparator;
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
	
		functionBody << statementIndent << "// for Space " << lps->getName() << statementSeparator;
		// if the current LPS is a subpartition then most of the fields of a thread Id can be 
		// copied from its parent LPU configuration
		if (lps->isSubpartitionSpace()) {
			functionBody << statementIndent << varName << ".groupId = 0" << statementSeparator;	
			functionBody << statementIndent << varName << ".ppuCount = 1" << statementSeparator;
			functionBody << statementIndent;
			functionBody  << varName << ".groupSize = ";
			functionBody << namePrefix << parentLps->getName() << "].groupSize";
			functionBody << statementSeparator;
			functionBody << statementIndent << varName << ".id = 0" << statementSeparator;
			functionBody << statementIndent;
			functionBody << "idsArray[Space_" << lps->getName() << "] = idsArray[Space_";
			functionBody << parentLps->getName() << "]" << statementSeparator << std::endl;
			continue;
		}

		// determine the total number of threads contributing in the parent PPS and current thread's 
		// index in that PPS 
		if (parent == mappingRoot) {
			functionBody << statementIndent << "threadCount = Total_Threads";
			functionBody << statementSeparator;
			groupThreadIdStr << "idsArray[Space_Root]";
		} else {
			functionBody << statementIndent;
			functionBody << "threadCount = " << namePrefix << parentLps->getName() << "].groupSize";
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

		// assign proper group Id, PPU count, and group size in the PPU-Ids variable created before 
		functionBody << statementIndent;
		functionBody  << varName << ".groupId = " << groupThreadIdStr.str() << " / groupSize";
		functionBody << statementSeparator;	
		functionBody << statementIndent;
		functionBody  << varName << ".ppuCount = " << partitionCount;
		functionBody << statementSeparator;
		functionBody << statementIndent;
		functionBody  << varName << ".groupSize = groupSize";
		functionBody << statementSeparator;

		// assign PPU Id to the thread depending on its groupThreadId
		functionBody << statementIndent;
		functionBody << "if (groupThreadId == 0) " << varName << ".id\n"; 
		functionBody << statementIndent << statementIndent << statementIndent;
		functionBody <<  "= " << varName << ".groupId";
		functionBody << statementSeparator;	
		functionBody << statementIndent;
		functionBody << "else " << varName << ".id = INVALID_ID";
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

void generateLpuDataStructures(const char *outputFile, MappingNode *mappingRoot) {
        
	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile;
        
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "Data structures representing LPS and LPU contents " << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}

	std::deque<MappingNode*> nodeQueue;
        nodeQueue.push_back(mappingRoot);
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		Space *lps = node->mappingConfig->LPS;
		List<const char*> *localArrays = lps->getLocallyUsedArrayNames();

		// create the object for containing references to data structures of the LPS
		programFile << "\nclass Space" << lps->getName() << "_Content {\n";
		programFile << "  public:\n";
		for (int i = 0; i < localArrays->NumElements(); i++) {
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(localArrays->Nth(i));
			ArrayType *arrayType = (ArrayType*) array->getType();
			const char *elemType = arrayType->getTerminalElementType()->getName();
			programFile << statementIndent << elemType << " *" << array->getName();
			programFile << statementSeparator;	
		}
		programFile << "};\n\n";

		// create the object for representing an LPU of the LPS
		programFile << "class Space" << lps->getName() << "_LPU : public LPU {\n";
		programFile << "  public:\n";
		for (int i = 0; i < localArrays->NumElements(); i++) {
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(localArrays->Nth(i));
			ArrayType *arrayType = (ArrayType*) array->getType();
			const char *elemType = arrayType->getTerminalElementType()->getName();
			programFile << statementIndent << elemType << " *" << array->getName();
			programFile << statementSeparator;
			int dimensions = array->getDimensionality();
			programFile << statementIndent << "PartitionDimension **";
			programFile << array->getName() << "PartDims";
			programFile << statementSeparator;	
		}
		programFile << "};\n";
	}
	
	programFile << std::endl;
	programFile.close();
}

void generateArrayMetadataAndEnvLinks(const char *outputFile, MappingNode *mappingRoot,
                List<EnvironmentLink*> *envLinks) {
	
	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile;
        
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "Data structures for Array-Metadata and Environment-Links " << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
	
	Space *rootLps = mappingRoot->mappingConfig->LPS;
	programFile << "\nclass ArrayMetadata {\n";
	programFile << "  public:\n";
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(localArrays->Nth(i));
		int dimensions = array->getDimensionality();
		programFile << statementIndent;
		programFile << "Dimension " << array->getName() << "Dims[" << dimensions << "]";
		programFile << statementSeparator;
	}
	programFile << "};\n";
	programFile << "ArrayMetadata arrayMetadata" << statementSeparator;
	
	programFile << "\nclass EnvironmentLinks {\n";
	programFile << "  public:\n";
	for (int i = 0; i < envLinks->NumElements(); i++) {
		EnvironmentLink *link = envLinks->Nth(i);
		if (!link->isExternal()) continue;
		const char *linkName = link->getVariable()->getName();
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(localArrays->Nth(i));
		ArrayType *arrayType = (ArrayType*) array->getType();
		const char *elemType = arrayType->getTerminalElementType()->getName();
		programFile << statementIndent << elemType << " *" << array->getName();
		programFile << statementSeparator;
		int dimensions = array->getDimensionality();
		programFile << statementIndent;
		programFile << "Dimension " << array->getName() << "Dims[" << dimensions << "]";
		programFile << statementSeparator;	
	}	
	programFile << "};\n";
	programFile << "EnvironmentLinks environmentLinks" << statementSeparator << std::endl;
}
