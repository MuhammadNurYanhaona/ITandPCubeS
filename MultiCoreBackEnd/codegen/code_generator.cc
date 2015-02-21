#include "code_generator.h"
#include "space_mapping.h"
#include "name_transformer.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../static-analysis/task_global.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <stdio.h>

void initializeOutputFiles(const char *headerFileName, 
		const char *programFileName, const char *initials) {

	std::string line;
        std::ifstream commIncludeFile("codegen/default-includes.txt");
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out);
        programFile.open (programFileName, std::ofstream::out);
        if (!programFile.is_open()) {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
	if (!headerFile.is_open()) {
		std::cout << "Unable to open output header file";
		std::exit(EXIT_FAILURE);
	}
	
	headerFile << "#ifndef _H_" << initials << std::endl;
	headerFile << "#define _H_" << initials << std::endl << std::endl;

	int taskNameIndex = string_utils::getLastIndexOf(headerFileName, '/') + 1;
	char *taskName = string_utils::substr(headerFileName, taskNameIndex, strlen(headerFileName));
	
	programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
        programFile << "header file for the task" << std::endl;
        programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	programFile << "#include \"" << taskName  << '"' << std::endl << std::endl;
                
	programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
        programFile << "header files included for different purposes" << std::endl;
        programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	
	if (commIncludeFile.is_open()) {
                while (std::getline(commIncludeFile, line)) {
			headerFile << line << std::endl;
			programFile << line << std::endl;
		}
		headerFile << std::endl;
		programFile << std::endl;
	} else {
		std::cout << "Unable to open common include file";
		std::exit(EXIT_FAILURE);
	}

	headerFile << "namespace " << string_utils::toLower(initials) << " {\n\n";
	programFile << "using namespace " << string_utils::toLower(initials) << ";\n\n";

	commIncludeFile.close();
	programFile.close();
	headerFile.close();
}

void generateThreadCountConstants(const char *outputFile,       
                MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig) {
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "constants for total and par core thread counts" << std::endl;
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

	// compute the total number of threads participating on each PPS starting from the root PPS
        nodeQueue.push_back(mappingRoot);
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		
		int threadCount = 1;
		PPS_Definition *pps = node->mappingConfig->PPS;
		Space *lps = node->mappingConfig->LPS;
		if (pps->id < highestUnpartitionedPpsId) {
			for (int i = 0; i < pcubesConfig->NumElements(); i++) {
				PPS_Definition *nextPps = pcubesConfig->Nth(i);
				if (nextPps->id >= highestUnpartitionedPpsId) continue;
				threadCount *= nextPps->units;
				if (pps == nextPps) break;
			}
		}
		programFile << "const int Space_" << lps->getName() << "_Threads = ";
		programFile << threadCount << ";" << std::endl;
	}	
	
	// compute the total number of threads that will participate in computing for the task
	int totalThreads = 1;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->id >= highestUnpartitionedPpsId) continue;
		totalThreads *= pps->units;
		if (pps->id == lowestPpsId) break;
	}
	programFile << "const int Total_Threads = " << totalThreads << ';' << std::endl;
	
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
	programFile << "const int Threads_Par_Core = " << threadsParCore << ';' << std::endl;
	programFile.close();
}

void generateFnForThreadIdsAllocation(const char *headerFileName, 
                const char *programFileName, 
                const char *initials,
                MappingNode *mappingRoot, 
                List<PPS_Definition*> *pcubesConfig) {

        std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile, headerFile;
        
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	headerFile << "\n/*-----------------------------------------------------------------------------------\n";
        headerFile << "function to generate PPU IDs and PPU group IDs for a thread\n";
        headerFile << "------------------------------------------------------------------------------------*/\n";
	programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "function to generate PPU IDs and PPU group IDs for a thread\n";
        programFile << "------------------------------------------------------------------------------------*/\n";

	std::ostringstream functionHeader;
        functionHeader << "getPpuIdsForThread(int threadNo)";
        std::ostringstream functionBody;
        
	functionBody << " {\n\n" << statementIndent;
	functionBody << "ThreadIds *threadIds = new ThreadIds";
	functionBody << statementSeparator;
	functionBody << statementIndent << "threadIds->threadNo = threadNo" << statementSeparator;
	functionBody << statementIndent << "threadIds->lpsCount = Space_Count" << statementSeparator;

	// allocate a new array to hold the PPU Ids of the thread
	functionBody << statementIndent<< "threadIds->ppuIds = new PPU_Ids[Space_Count]" << statementSeparator;
	// declare a local array to hold the index of the thread in different PPS group for ID assignment
	// to be done accurately 
	functionBody << statementIndent << "int idsArray[Space_Count]" << statementSeparator;
	functionBody << statementIndent << "idsArray[Space_Root] = threadNo" << statementSeparator; 

	// enter default values for the root LPS
	std::ostringstream rootVarStr;
	rootVarStr << "threadIds->ppuIds[Space_";
	rootVarStr << mappingRoot->mappingConfig->LPS->getName() << "]";
	std::string rootName = rootVarStr.str();
	functionBody << std::endl << statementIndent << "// for Space Root\n";
	functionBody << statementIndent << rootName << ".lpsName = \"Root\"" << statementSeparator;
	functionBody << statementIndent << rootName << ".groupId = 0" << statementSeparator;
	functionBody << statementIndent << rootName << ".groupSize = Total_Threads" << statementSeparator;
	functionBody << statementIndent << rootName << ".ppuCount = 1" << statementSeparator;
	functionBody << statementIndent << rootName << ".id = (threadNo == 0) ? 0 : INVALID_ID";
	functionBody << statementSeparator;
	
	// then begin processing for other LPSes
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
		functionBody << statementIndent << varName << ".lpsName = \"" << lps->getName();
		functionBody << "\"" << statementSeparator;

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

	headerFile << "ThreadIds *" << functionHeader.str() << ";\n\n";	
	programFile << std::endl << "ThreadIds *" << initials << "::"; 
	programFile <<functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

void generateLpuDataStructures(const char *outputFile, MappingNode *mappingRoot) {
       
	std::cout << "Generating data structures for LPUs\n";
 
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
		programFile << "};\n";
		
		// declare a global reference of this object to be used during LPU generation
		programFile << "Space" << lps->getName() << "_Content space" << lps->getName() << "Content";
		programFile << statementSeparator << std::endl;

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
			programFile << statementIndent << "PartDimension ";
			programFile << array->getName() << "PartDims[" << array->getDimensionality() << "]";
			programFile << statementSeparator;	
		}
		// add a specific lpu_id static array with dimensionality equals to the dimensions of the LPS
		if (lps->getDimensionCount() > 0) {
			programFile << statementIndent << "int lpuId[";
			programFile << lps->getDimensionCount() << "]";
			programFile << statementSeparator;
		}
		// define a print function for the LPU
		programFile << std::endl;
		programFile << statementIndent << "void print(std::ofstream &stream, int indent)" << statementSeparator;	
		programFile << "};\n";
	}
	
	programFile << std::endl;
	programFile.close();
}

void generatePrintFnForLpuDataStructures(const char *initials, const char *outputFile, MappingNode *mappingRoot) {

	std::cout << "Generating print functions for LPUs\n";
	
	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile;
        
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "Print functions for LPUs " << std::endl;
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
		
		programFile << std::endl;
		programFile << "void " << initials << "::Space" << lps->getName() << "_LPU::print";
		programFile << "(std::ofstream &stream, int indentLevel) {\n";
		
		List<const char*> *localArrays = lps->getLocallyUsedArrayNames();
		for (int i = 0; i < localArrays->NumElements(); i++) {
			const char *arrayName = localArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(arrayName);
			int dimensions = array->getDimensionality();
			programFile << statementIndent << "for (int i = 0; i < indentLevel; i++) ";
			programFile << "stream << '\\t'" << statementSeparator;
			programFile << statementIndent << "stream << \"Array: " << arrayName << "\"";
			programFile << " << std::endl";
			programFile << statementSeparator;
			for (int j = 0; j < dimensions; j++) {
				programFile << statementIndent;
				programFile << arrayName << "PartDims[" << j;
				programFile  << "].print(stream, indentLevel + 1)";
				programFile << statementSeparator;
			}
		}
		programFile << statementIndent << "stream.flush()" << statementSeparator;
		programFile << "}\n";
	}

	programFile << std::endl;
	programFile.close();
}

void generateFnToInitiateRootLPSContent(const char *headerFileName, const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot,
                List<const char*> *externalEnvLinks) {

	std::cout << "Generating functions for initializing LPS contents\n";

        std::string stmtSeparator = ";\n";
        std::string stmtIndent = "\t";
	std::string paramSeparator = ", ";
	std::ofstream programFile, headerFile;
        
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	headerFile << "\n/*-----------------------------------------------------------------------------------\n";
        headerFile << "function to initialize the content reference objects of LPSes\n";
        headerFile << "------------------------------------------------------------------------------------*/\n";
	programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "function to initialize the content reference objects of LPSes\n";
        programFile << "------------------------------------------------------------------------------------*/\n";

	Space *rootLps = mappingRoot->mappingConfig->LPS;

	std::string functionHeader;
        functionHeader = "initializeRootLPSContent(EnvironmentLinks *envLinks, ArrayMetadata *metadata)";
	std::ostringstream lpsVarStr;
	lpsVarStr << "space" << rootLps->getName() << "Content";
	std::string lpsVar = lpsVarStr.str();
        
	std::ostringstream functionBody;
	functionBody << "{\n";
	// Iterate over all the arrays that can be found in the root LPS. Then allocate or initialize them. Note that 
	// this is an unoptimized implementation. TODO in the optimized case, we should only allocate variable in 
	// LPSes where they are needed
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		const char *arrayName = localArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(arrayName);
		ArrayType *arrayType = (ArrayType*) array->getType();
		Type *elemType = arrayType->getTerminalElementType();
		bool isInEnvLinks = false;
		for (int j = 0; j < externalEnvLinks->NumElements(); j++) {
			if (strcmp(arrayName, externalEnvLinks->Nth(j)) == 0) {
				isInEnvLinks = true;
				break;
			}
		}
		// if the variable in environment links then copy its reference from there
		if (isInEnvLinks) {
			functionBody << stmtIndent;
			functionBody << lpsVar << "." << arrayName << " = envLinks->" << arrayName;
			functionBody << stmtSeparator;
		// otherwise allocate an array for the variable
		} else {
			int dimensionCount = array->getDimensionality();
			functionBody << stmtIndent;
			functionBody << lpsVar << "." << arrayName << " = allocate::allocateArray ";
			functionBody << "<" << elemType->getName() << "> (";
			functionBody << dimensionCount << paramSeparator;
			functionBody << "metadata->" << arrayName << "Dims)";
			functionBody << stmtSeparator;
			
			// if the array contains primitive type objects then zero fill it
			ListType *list = dynamic_cast<ListType*>(elemType);
        		MapType *map = dynamic_cast<MapType*>(elemType);
        		NamedType *object = dynamic_cast<NamedType*>(elemType);
			if (list == NULL && map == NULL && object == NULL) {
				functionBody << stmtIndent;
				functionBody << "allocate::zeroFillArray ";
				functionBody << "<" << elemType->getName() << "> (0";
				functionBody << paramSeparator << lpsVar << "." << arrayName;
				functionBody << paramSeparator << dimensionCount << paramSeparator;
				functionBody << "metadata->" << arrayName << "Dims)";
				functionBody << stmtSeparator;
			}
		}
		// mark the variable to be allocated in the root LPS
		array->getUsageStat()->flagAllocated();
	}	
	functionBody << "}\n";

	headerFile << "void " << functionHeader << ";\n";	
	programFile << std::endl << "void " << initials << "::"; 
	programFile <<functionHeader << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

void generateFnToInitiateLPSesContent(const char *headerFileName, const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot) {
        
	std::string stmtSeparator = ";\n";
        std::string stmtIndent = "\t";
	std::string paramSeparator = ", ";
	std::ofstream programFile, headerFile;
        
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	std::string functionHeader;
        functionHeader = "initializeLPSesContents(ArrayMetadata *metadata)";
	
	std::ostringstream functionBody;
	functionBody << "{\n";

	// Since Root LPS content is initialized elsewhere we start the computation from immediate children
	// of the Root LPS
	std::deque<MappingNode*> nodeQueue;
        for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
        	nodeQueue.push_back(mappingRoot->children->Nth(i));
        }

	// Until the queue is empty get information about LPSes one by one in order and initialize their contents
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		// get the LPS reference from the mapping configuration
		Space *lps = node->mappingConfig->LPS;

		// get the list of data structures for current LPS
		List<const char*> *localArrays = lps->getLocallyUsedArrayNames();
		
		// filter the list and keep only those structures that are been accessed in the current LPS
		List<const char*> *filteredArrays = new List<const char*>;
		for (int i = 0; i < localArrays->NumElements(); i++) {
			const char *arrayName = localArrays->Nth(i);
			DataStructure *structure = lps->getLocalStructure(arrayName);
			if (structure->getUsageStat()->isAccessed()) {
				filteredArrays->Append(arrayName);
			}
		} 

		// populate a map of parent LPSes for different data structures indicating where the structure is 
		// last been allocated.
		Hashtable<Space*> *allocatingParentMap = new Hashtable<Space*>;
		for (int i = 0; i < filteredArrays->NumElements(); i++) {
			const char *arrayName = filteredArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(arrayName);
			Space *allocatingParentLps = mappingRoot->mappingConfig->LPS;
			Space *currentLps =  lps;
			while ((currentLps = currentLps->getParent()) != NULL) {
				DataStructure *structure = currentLps->getLocalStructure(arrayName);
				if (structure == NULL) continue;
				if (structure->getUsageStat()->isAllocated()) {
					allocatingParentLps = currentLps;
					break;
				}
			}
			allocatingParentMap->Enter(arrayName, allocatingParentLps, true);
		}

		// now we have sufficient information about allocating data structures or copying references to them
		// start by writing a comment indicating what LPS we are handling at this moment
		functionBody << stmtIndent << "//Processing Space " << lps->getName() << " contents\n";
	
		// get the variable name for the content references of current LPS
		std::ostringstream lpsVarStr;
		lpsVarStr << "space" << lps->getName() << "Content";
		std::string lpsVar = lpsVarStr.str();

		// iterate over the accessed data structure list
		for (int i = 0; i < filteredArrays->NumElements(); i++) {
			const char *arrayName = filteredArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(arrayName);
			Space *allocatingParent = allocatingParentMap->Lookup(arrayName);
			
			// if the data structure is used multiple time in the LPS and been reordered from its earlier
			// LPSes content then it should be allocated in the current LPS
			if (array->getUsageStat()->isMultipleAccess() && array->isReordered(allocatingParent)) {
				
				int dimensionCount = array->getDimensionality();
				ArrayType *arrayType = (ArrayType*) array->getType();
				Type *elemType = arrayType->getTerminalElementType();
				
				functionBody << stmtIndent;
				functionBody << lpsVar << "." << arrayName << " = allocate::allocateArray ";
				functionBody << "<" << elemType->getName() << "> (";
				functionBody << dimensionCount << paramSeparator;
				functionBody << "metadata->" << arrayName << "Dims)";
				functionBody << stmtSeparator;			
			
				// if the array contains primitive type objects then zero fill it
				ListType *list = dynamic_cast<ListType*>(elemType);
        			MapType *map = dynamic_cast<MapType*>(elemType);
        			NamedType *object = dynamic_cast<NamedType*>(elemType);
				if (list == NULL && map == NULL && object == NULL) {
					functionBody << stmtIndent;
					functionBody << "allocate::zeroFillArray ";
					functionBody << "<" << elemType->getName() << "> (0";
					functionBody << paramSeparator << lpsVar << "." << arrayName;
					functionBody << paramSeparator << dimensionCount << paramSeparator;
					functionBody << "metadata->" << arrayName << "Dims)";
					functionBody << stmtSeparator;
				}

				// flag the array as been allocated in current LPS
				array->getUsageStat()->flagAllocated();
				
			// otherwise it should get the reference from the allocating parent
			} else {
				functionBody << stmtIndent;
				functionBody << lpsVar << "." << arrayName  << " = ";
				functionBody << "space" << allocatingParent->getName() << "Content";
				functionBody << "." << arrayName;
				functionBody << stmtSeparator;
			}
		}
	}	

	functionBody << "}\n";

	headerFile << "void " << functionHeader << ";\n";	
	programFile << "void " << initials << "::"; 
	programFile <<functionHeader << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

void generateFnToInitiateLPSesContentSimple(const char *headerFileName, const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot) {
        
	std::string stmtSeparator = ";\n";
        std::string stmtIndent = "\t";
	std::string paramSeparator = ", ";
	std::ofstream programFile, headerFile;
        
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	std::string functionHeader;
        functionHeader = "initializeLPSesContents(ArrayMetadata *metadata)";
	
	std::ostringstream functionBody;
	functionBody << "{\n";

	// Since Root LPS content is initialized elsewhere we start the computation from immediate children
	// of the Root LPS
	std::deque<MappingNode*> nodeQueue;
        for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
        	nodeQueue.push_back(mappingRoot->children->Nth(i));
        }

	// Until the queue is empty get information about LPSes one by one in order and initialize their contents
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		// get the LPS reference from the mapping configuration
		Space *lps = node->mappingConfig->LPS;

		// get the list of data structures for current LPS
		List<const char*> *localArrays = lps->getLocallyUsedArrayNames();
		
		// filter the list and keep only those structures that are been accessed in the current LPS
		List<const char*> *filteredArrays = new List<const char*>;
		for (int i = 0; i < localArrays->NumElements(); i++) {
			const char *arrayName = localArrays->Nth(i);
			DataStructure *structure = lps->getLocalStructure(arrayName);
			if (structure->getUsageStat()->isAccessed()) {
				filteredArrays->Append(arrayName);
			}
		} 

		// populate a map of parent LPSes for different data structures indicating where the structure is 
		// last been allocated.
		Hashtable<Space*> *allocatingParentMap = new Hashtable<Space*>;
		for (int i = 0; i < filteredArrays->NumElements(); i++) {
			const char *arrayName = filteredArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(arrayName);
			Space *allocatingParentLps = mappingRoot->mappingConfig->LPS;
			Space *currentLps =  lps;
			while ((currentLps = currentLps->getParent()) != NULL) {
				DataStructure *structure = currentLps->getLocalStructure(arrayName);
				if (structure == NULL) continue;
				if (structure->getUsageStat()->isAllocated()) {
					allocatingParentLps = currentLps;
					break;
				}
			}
			allocatingParentMap->Enter(arrayName, allocatingParentLps, true);
		}

		// now we have sufficient information about allocating data structures or copying references to them
		// start by writing a comment indicating what LPS we are handling at this moment
		functionBody << stmtIndent << "//Processing Space " << lps->getName() << " contents\n";
	
		// get the variable name for the content references of current LPS
		std::ostringstream lpsVarStr;
		lpsVarStr << "space" << lps->getName() << "Content";
		std::string lpsVar = lpsVarStr.str();

		// iterate over the accessed data structure list
		for (int i = 0; i < filteredArrays->NumElements(); i++) {
			const char *arrayName = filteredArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(arrayName);
			Space *allocatingParent = allocatingParentMap->Lookup(arrayName);
			functionBody << stmtIndent;
			functionBody << lpsVar << "." << arrayName  << " = ";
			functionBody << "space" << allocatingParent->getName() << "Content";
			functionBody << "." << arrayName;
			functionBody << stmtSeparator;
		}
	}	

	functionBody << "}\n";

	headerFile << "void " << functionHeader << ";\n";	
	programFile << "void " << initials << "::"; 
	programFile <<functionHeader << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

List<const char*> *generateArrayMetadataAndEnvLinks(const char *outputFile, MappingNode *mappingRoot,
                List<EnvironmentLink*> *envLinks) {

	std::cout << "Generating array metadata and environment links\n";
	
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
	
	// construct an array metadata object by listing all arrays present in the root LPS
	Space *rootLps = mappingRoot->mappingConfig->LPS;
	programFile << "\nclass ArrayMetadata : public Metadata {\n";
	programFile << "  public:\n";
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(localArrays->Nth(i));
		int dimensions = array->getDimensionality();
		programFile << statementIndent;
		programFile << "Dimension " << array->getName() << "Dims[" << dimensions << "]";
		programFile << statementSeparator;
	}
	programFile << std::endl;
	programFile << statementIndent << "ArrayMetadata()" << statementSeparator;
	programFile << statementIndent << "void print(std::ofstream &stream)" << statementSeparator;
	programFile << "};\n";
	programFile << "ArrayMetadata arrayMetadata" << statementSeparator;
	
	// create a class for environment links; also generate a list of the name of such links to be returned
	List<const char*> *linkList = new List<const char*>;
	programFile << "\nclass EnvironmentLinks {\n";
	programFile << "  public:\n";
	for (int i = 0; i < envLinks->NumElements(); i++) {
		EnvironmentLink *link = envLinks->Nth(i);
		if (!link->isExternal()) continue;
		const char *linkName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getLocalStructure(linkName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) {
               		ArrayType *arrayType = (ArrayType*) array->getType();
               		const char *elemType = arrayType->getTerminalElementType()->getName();
               		programFile << statementIndent << elemType << " *" << array->getName();
               		programFile << statementSeparator;
               		int dimensions = array->getDimensionality();
               		programFile << statementIndent;
               		programFile << "Dimension " << array->getName() << "Dims[" << dimensions << "]";
               		programFile << statementSeparator;
		} else {
			Type *type = structure->getType();
			const char *declaration = type->getCppDeclaration(structure->getName());
			programFile << statementIndent << declaration << statementSeparator;
		}
		linkList->Append(linkName);
	}
	programFile << std::endl;	
	programFile << statementIndent << "void print(std::ofstream &stream)" << statementSeparator;
	programFile << "};\n";
	programFile << "EnvironmentLinks environmentLinks" << statementSeparator << std::endl;
	programFile.close();
	return linkList;
}

void generateFnForMetadataAndEnvLinks(const char *taskName, const char *initials, 
		const char *outputFile, MappingNode *mappingRoot,
                List<const char*> *externalLinks) {

	std::cout << "Generating function implementations for array metadata and environment links\n";
	
	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::ofstream programFile;
        
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "Functions for ArrayMetadata and EnvironmentLinks " << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
	
	Space *rootLps = mappingRoot->mappingConfig->LPS;

	// generate constructor for array metadata 
	programFile << std::endl << initials << "::ArrayMetadata::ArrayMetadata() : Metadata() {\n";
	programFile << statementIndent << "setTaskName";
	programFile << "(\"" << taskName << "\")" << statementSeparator;  
	programFile << "}" << std::endl << std::endl;

	// generate a print function for array metadata
	programFile << "void " << initials << "::ArrayMetadata::" << "print(std::ofstream &stream) {\n";
	programFile << statementIndent << "stream << \"Array Metadata\" << std::endl" << statementSeparator;
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		const char *arrayName = localArrays->Nth(i);
		programFile << statementIndent << "stream << \"Array: " << arrayName << "\"";
		programFile << statementSeparator;
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(arrayName);
		int dimensions = array->getDimensionality();
		for (int j = 0; j < dimensions; j++) {
			programFile << statementIndent << "stream << ' '" << statementSeparator;
			programFile << statementIndent;
			programFile << arrayName << "Dims[" << j << "].print(stream)";
			programFile << statementSeparator;
		}
		programFile << statementIndent << "stream << std::endl" << statementSeparator;
	}
	programFile << statementIndent << "stream.flush()" << statementSeparator;
	programFile << "}" << std::endl << std::endl;
	
	programFile.close();
}

void closeNameSpace(const char *headerFile) {
	std::ofstream programFile;
	programFile.open (headerFile, std::ofstream::app);
	if (programFile.is_open()) {
		programFile << std::endl << '}' << std::endl;
		programFile << "#endif" << std::endl;
		programFile.close();
	} else {
		std::cout << "Could not open header file" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	programFile.close();
}

void generateClassesForTuples(const char *filePath, List<TupleDef*> *tupleDefList) {
	std::ofstream headerFile;
	headerFile.open(filePath, std::ofstream::out);
	if (!headerFile.is_open()) {
		std::cout << "Unable to open header file for tuple definitions\n";
		std::exit(EXIT_FAILURE);
	}
	headerFile << "#ifndef _H_tuple\n";
	headerFile << "#define _H_tuple\n\n";

	// by default include header file for standard vector for any list variable that may present
	// in any tuple definition
	headerFile << "#include <iostream>\n";	
	headerFile << "#include <vector>\n\n";	

	// first have a list of forward declarations for all tuples to avoid having errors during 
	// compilation of individual classes
	for (int i = 0; i < tupleDefList->NumElements(); i++) {
		TupleDef *tupleDef = tupleDefList->Nth(i);
		headerFile << "class " << tupleDef->getId()->getName() << ";\n";
	}
	headerFile << "\n";

	// then generate a class for each tuple in the list
	for (int i = 0; i < tupleDefList->NumElements(); i++) {
		// if the tuple definition has no element inside then ignore it and proceed to the next
		TupleDef *tupleDef = tupleDefList->Nth(i);
		List<VariableDef*> *variables = tupleDef->getComponents();
		// otherwise, generate a new class and add the elements as public components
		headerFile << "class " << tupleDef->getId()->getName() << " {\n";
		headerFile << "  public:\n";
		for (int j = 0; j < variables->NumElements(); j++) {
			headerFile << "\t";
			VariableDef *variable = variables->Nth(j);
			Type *type = variable->getType();
			const char *varName = variable->getId()->getName();
			headerFile << type->getCppDeclaration(varName);
			headerFile << ";\n";
		}
		headerFile << "};\n\n";
	}

	headerFile << "#endif\n";
	headerFile.close();
}

void generateClassesForGlobalScalars(const char *filePath, List<TaskGlobalScalar*> *globalList) {
	
	std::cout << "Generating structures holding task global and thread local scalar\n";

	std::ofstream headerFile;
	headerFile.open (filePath, std::ofstream::out | std::ofstream::app);
	if (!headerFile.is_open()) {
		std::cout << "Unable to open output header file for task\n";
		std::exit(EXIT_FAILURE);
	}
                
	headerFile << "/*-----------------------------------------------------------------------------------\n";
        headerFile << "Data structures for Task-Global and Thread-Local scalar variables\n";
        headerFile << "------------------------------------------------------------------------------------*/\n\n";
	
	std::ostringstream taskGlobals, threadLocals;
	taskGlobals << "class TaskGlobals {\n";
	taskGlobals << "  public:\n";
	threadLocals << "class ThreadLocals {\n";
	threadLocals << "  public:\n";

	for (int i = 0; i < globalList->NumElements(); i++) {
		TaskGlobalScalar *scalar = globalList->Nth(i);
		// determine to which class the global should go into
		std::ostringstream *stream = &taskGlobals;
		if (scalar->isLocallyManageable()) {
			stream = &threadLocals;
		}
		// then write the variable declaration within the stream
		Type *type = scalar->getType();
		*stream << "\t";
		*stream << type->getCppDeclaration(scalar->getName());
		*stream << ";\n";		
	}
	
	taskGlobals << "};\n\n";
	threadLocals << "};\n";

	headerFile << taskGlobals.str() << threadLocals.str();
	headerFile.close();
}

void generateInitializeFunction(const char *headerFileName, const char *programFileName, const char *initials,
                List<const char*> *envLinkList, TaskDef *taskDef, Space *rootLps) {
        
	std::cout << "Generating function for the initialize block\n";

	std::string statementSeparator = ";\n";
        std::string statementIndent = "\t";
	std::string parameterSeparator = ", ";
	std::ofstream programFile, headerFile;
        
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file for initialize block generation";
		std::exit(EXIT_FAILURE);
	}
                
	headerFile << "\n/*-----------------------------------------------------------------------------------\n";
        headerFile << "function for the initialize block\n";
        headerFile << "------------------------------------------------------------------------------------*/\n";
	programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "function for the initialize block\n";
        programFile << "------------------------------------------------------------------------------------*/\n";

	// put five default parameters for metadata, env-Links, task-globals, thread-locals, and partition 
	// configuration
	std::ostringstream functionHeader;
        functionHeader << "initializeTask(ArrayMetadata *arrayMetadata";
	functionHeader << parameterSeparator << '\n' << statementIndent << statementIndent; 
        functionHeader << "EnvironmentLinks environmentLinks";
	functionHeader << parameterSeparator << '\n' << statementIndent << statementIndent; 
        functionHeader << "TaskGlobals *taskGlobals";
	functionHeader << parameterSeparator << '\n' << statementIndent << statementIndent; 
	functionHeader << "ThreadLocals *threadLocals";
	functionHeader << parameterSeparator << '\n' << statementIndent << statementIndent;
	functionHeader << string_utils::getInitials(taskDef->getName());
	functionHeader << "Partition partition";

        std::ostringstream functionBody;
	functionBody << "{\n\n";

	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		const char *envLink = envLinkList->Nth(i);
		// if the variable is an array then its dimension information needs to be copied from the
		// environment link object to array metadata object of all subsequent references
		if (transformer->isGlobalArray(envLink)) {
			const char *varName = transformer->getTransformedName(envLink, true, false);
			ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(envLink);
			int dimensionCount = array->getDimensionality();
			for (int j = 0; j < dimensionCount; j++) {
				functionBody << statementIndent;
				functionBody << varName << "[" << j << "]";
				functionBody << " = " << "environmentLinks.";
				functionBody << envLink;
				functionBody << "Dims[" << j << "]";
				functionBody << statementSeparator;
			}
		// otherwise the value of the scalar variable should be copied back to task global or thread
		// local variable depending on what is the right destination
		} else {
			functionBody << statementIndent;
			functionBody << transformer->getTransformedName(envLink, true, false);
			functionBody << " = " << "environmentLinks.";
			functionBody << envLink;
			functionBody << statementSeparator;
		}
	}

	InitializeInstr *initSection = taskDef->getInitSection();
	if (initSection != NULL) {
		
		// iterate over all initialization parameters and add them as function arguments
		List<const char*> *argNames = initSection->getArguments();
		List<Type*> *argTypes = initSection->getArgumentTypes();
		for (int i = 0; i < argNames->NumElements(); i++) {
			const char *arg = argNames->Nth(i);
			Type *type = argTypes->Nth(i);
			functionHeader << parameterSeparator;
			functionHeader << "\n" << statementIndent << statementIndent;
			functionHeader << type->getCppDeclaration(arg);
			// if any argument matches a global variable in the task then copy it to the appropriate
			// data structure
			if (transformer->isThreadLocal(arg) || transformer->isTaskGlobal(arg)) {
				functionBody << statementIndent;
				functionBody << transformer->getTransformedName(arg, true, false);
				functionBody << " = " << arg;
				functionBody << statementSeparator;
			}
		}
		// then translate the user code in the init section into a c++ instruction stream
		initSection->generateCode(functionBody);
	}

	functionHeader << ")";
	functionBody << "}\n";

	headerFile << "void " << functionHeader.str() << ";\n\n";	
	programFile << std::endl << "void " << initials << "::"; 
	programFile <<functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}
