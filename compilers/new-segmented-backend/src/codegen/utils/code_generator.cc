#include "code_generator.h"
#include "space_mapping.h"
#include "name_transformer.h"
#include "code_constant.h"
#include "task_global.h"

#include "../../../../frontend/src/syntax/ast_def.h"
#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/semantics/task_space.h"
#include "../../../../frontend/src/static-analysis/reduction_info.h"
#include "../../../../frontend/src/codegen-helper/extern_config.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"
#include "../../../../common-libs/utils/common_utils.h"
#include "../../../../common-libs/utils/decorator_utils.h"
#include "../../../../common-libs/domain-obj/constant.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <stdio.h>

void initializeOutputFiles(const char *headerFileName, 
		const char *programFileName, const char *initials, TaskDef *taskDef) {

	std::string line;
        std::ifstream commIncludeFile("config/default-includes.txt");
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
	
	decorator::writeSectionHeader(programFile, "header file for the task");
	programFile << "#include \"" << taskName  << '"' << std::endl << std::endl;
        decorator::writeSectionHeader(programFile, "header files for different purposes");        
	
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

	// Since we are generating C++ code any external code block written in C and C++ can be directly
	// placed within the generated code but we have to include the proper header files in the generated
	// program to make this scheme work. So here we are including those headers.
	IncludesAndLinksMap *externConfig = taskDef->getExternBlocksHeadersAndLibraries();
	List<const char*> *headerIncludes = new List<const char*>;
        if (externConfig->hasExternBlocksForLanguage("C++")) {
                LanguageIncludesAndLinks *cPlusHeaderAndLinks
                                = externConfig->getIncludesAndLinksForLanguage("C++");
		string_utils::combineLists(headerIncludes, cPlusHeaderAndLinks->getHeaderIncludes());	
        }
	if (externConfig->hasExternBlocksForLanguage("C")) {
		LanguageIncludesAndLinks *cHeaderAndLinks
                                = externConfig->getIncludesAndLinksForLanguage("C");
                string_utils::combineLists(headerIncludes, cHeaderAndLinks->getHeaderIncludes());
	}
	if (headerIncludes->NumElements() > 0) {
		programFile << "// header files needed to execute external code blocks\n";
		for (int i = 0; i < headerIncludes->NumElements(); i++) {
			programFile << "#include ";
			const char *headerFile = headerIncludes->Nth(i);
			if (headerFile[0] == '"') {
				programFile << headerFile << '\n';
			} else {
				programFile << '<' << headerFile << '>' << '\n';
			}
		}
		programFile << '\n';
	}

	headerFile << "namespace " << string_utils::toLower(initials) << " {\n\n";
	programFile << "using namespace " << string_utils::toLower(initials) << ";\n\n";

	commIncludeFile.close();
	programFile.close();
	headerFile.close();
}

void generateThreadCountConstants(const char *outputFile, MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig) {
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
		const char *header = "thread count constants";
		decorator::writeSectionHeader(programFile, header);
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

	const char *sysConstMsg = "System Constants";
	decorator::writeSubsectionHeader(programFile, sysConstMsg);
	
	// compute the total number of threads participating on each PPS starting from the root PPS
	List<int> *lpsThreadCounts = new List<int>;
	List<const char*> *lpsNames = new List<const char*>;
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
		programFile << "const int Max_Space_" << lps->getName() << "_Threads = ";
		programFile << threadCount << stmtSeparator;
		lpsNames->Append(lps->getName());
		lpsThreadCounts->Append(threadCount);
	}	
	
	// compute the total number of threads that will participate in computing for the task
	int totalThreads = 1;
	if (highestUnpartitionedPpsId > lowestPpsId) {
		for (int i = 0; i < pcubesConfig->NumElements(); i++) {
			PPS_Definition *pps = pcubesConfig->Nth(i);
			if (pps->id >= highestUnpartitionedPpsId) continue;
			totalThreads *= pps->units;
			if (pps->id == lowestPpsId) break;
		}
	}
	programFile << "const int Max_Total_Threads = " << totalThreads << stmtSeparator;
	
	// The actual total threads count is a static variable as opposed to a constant like others. 
	// This is because its value is updated by the task executor function at runtime depending
	// on the number of actual segments that do the task's computation. 
	programFile << "static int Total_Threads = " << totalThreads << stmtSeparator;

	const char *sgConstMsg = "Segment Constants";
	decorator::writeSubsectionHeader(programFile, sgConstMsg);
	
	// determine how many threads can operate within each memory segment
	int segmentedPpsIndex = 0;
	bool segmentationFound = false;
	while (true) {
		if (segmentedPpsIndex >= pcubesConfig->NumElements()) break;
		if (pcubesConfig->Nth(segmentedPpsIndex)->segmented == true) {
			segmentationFound = true;
			break;
		}
		segmentedPpsIndex++;
	}
	if (!segmentationFound) segmentedPpsIndex = 0;
	int threadsPerSegment = 1;
	int segmentedPpsId = pcubesConfig->Nth(segmentedPpsIndex)->id;
	if (highestUnpartitionedPpsId > lowestPpsId && segmentedPpsId > lowestPpsId) {
		for (int i = segmentedPpsIndex + 1; i < pcubesConfig->NumElements(); i++) {
			PPS_Definition *pps = pcubesConfig->Nth(i);
			if (pps->id >= highestUnpartitionedPpsId) continue;
			threadsPerSegment *= pps->units;
			if (pps->id == lowestPpsId) break;
		}
	}
	programFile << "const int Threads_Per_Segment = " << threadsPerSegment << stmtSeparator;
	int totalSegments = totalThreads / threadsPerSegment;
	programFile << "const int Max_Segments_Count = " << totalSegments << stmtSeparator;
	
	// determine the total number of threads per segment for different LPSes
	for (int i = 0; i < lpsNames->NumElements(); i++) {
		const char *lpsName = lpsNames->Nth(i);
		int lpsThreads = lpsThreadCounts->Nth(i);
		int lpsThreadsInSegment = std::max(1, lpsThreads / totalSegments);
		programFile << "const int Space_" << lpsName << "_Threads_Per_Segment = ";
		programFile << lpsThreadsInSegment << stmtSeparator;
	} 

	const char *hwConstMsg = "Hardware Constants";
	decorator::writeSubsectionHeader(programFile, hwConstMsg);
	
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
	programFile << "const int Threads_Per_Core = " << threadsParCore << stmtSeparator;

	// calculate where the hardware unit boundary lies in the PCubeS hierarchy so that processor numbering
	// can be reset at proper interval
	int processorsPerPhyUnit = 1;
	bool phyUnitFound = false;
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (!phyUnitFound && !pps->physicalUnit) continue;
		if (pps->physicalUnit) {
			phyUnitFound = true;
			continue;
		}
		processorsPerPhyUnit *= pps->units;
	}
	programFile << "const int Processors_Per_Phy_Unit = ";	
	if (phyUnitFound) programFile << processorsPerPhyUnit << stmtSeparator;
	else programFile << totalThreads << stmtSeparator;

	// If the lowest LPS is mapped to a PPS above the core space then threads should be more apart than
	// 1 core space processor. In that case we need to determine how far we should jump as we assign 
	// threads to processors. To aid in that calculation we need to calculate another constant. We call 
	// core jump.
        int lastMappedPpsId = mappingRoot->mappingConfig->PPS->id;
	nodeQueue.push_back(mappingRoot);
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		PPS_Definition *pps = node->mappingConfig->PPS;
		if (pps->id < lastMappedPpsId) {
			lastMappedPpsId = pps->id;
		}
	}
	int coreJump = 1;
	if (lastMappedPpsId > coreSpaceId) {
		for (int i = 0; i < pcubesConfig->NumElements(); i++) {
			PPS_Definition *pps = pcubesConfig->Nth(i);
			if (pps->id >= lastMappedPpsId) continue;
			coreJump *= pps->units;
			if (pps->id == coreSpaceId) break;
		}
	}
	programFile << "const int Core_Jump = " << coreJump << stmtSeparator;

	programFile.close();
}

void generateFnForThreadIdsAllocation(const char *headerFileName, 
                const char *programFileName, 
                const char *initials,
                MappingNode *mappingRoot, 
                List<PPS_Definition*> *pcubesConfig) {

	std::ofstream programFile, headerFile;
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	const char *message = "functions to generate PPU IDs and PPU group IDs for a thread";
	decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
	decorator::writeSectionHeader(programFile, message);

	std::ostringstream functionHeader;
        functionHeader << "getPpuIdsForThread(int threadNo)";
        std::ostringstream functionBody;
        
	functionBody << " {\n\n" << indent;
	functionBody << "ThreadIds *threadIds = new ThreadIds";
	functionBody << stmtSeparator;
	functionBody << indent << "threadIds->threadNo = threadNo" << stmtSeparator;
	functionBody << indent << "threadIds->lpsCount = Space_Count" << stmtSeparator;

	// allocate a new array to hold the PPU Ids of the thread
	functionBody << indent<< "threadIds->ppuIds = new PPU_Ids[Space_Count]" << stmtSeparator;
	// declare a local array to hold the index of the thread in different PPS group for ID assignment
	// to be done accurately 
	functionBody << indent << "int idsArray[Space_Count]" << stmtSeparator;
	functionBody << indent << "idsArray[Space_Root] = threadNo" << stmtSeparator; 

	// enter default values for the root LPS
	std::ostringstream rootVarStr;
	rootVarStr << "threadIds->ppuIds[Space_";
	rootVarStr << mappingRoot->mappingConfig->LPS->getName() << "]";
	std::string rootName = rootVarStr.str();
	functionBody << std::endl << indent << "// for Space Root\n";
	functionBody << indent << rootName << ".lpsName = \"Root\"" << stmtSeparator;
	functionBody << indent << rootName << ".groupId = 0" << stmtSeparator;
	functionBody << indent << rootName << ".groupSize = Total_Threads" << stmtSeparator;
	functionBody << indent << rootName << ".ppuCount = 1" << stmtSeparator;
	functionBody << indent << rootName << ".id = (threadNo == 0) ? 0 : INVALID_ID";
	functionBody << stmtSeparator;
	
	// then begin processing for other LPSes
	std::deque<MappingNode*> nodeQueue;
        for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
        	nodeQueue.push_back(mappingRoot->children->Nth(i));
        }

	// declare some local variables needed for thread Id calculation
	functionBody << std::endl;
	functionBody << indent << "int threadCount" << stmtSeparator;
	functionBody << indent << "int groupSize" << stmtSeparator;
	functionBody << indent << "int groupThreadId" << stmtSeparator;
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
	
		functionBody << indent << "// for Space " << lps->getName() << stmtSeparator;
		functionBody << indent << varName << ".lpsName = \"" << lps->getName();
		functionBody << "\"" << stmtSeparator;

		// if the current LPS is a subpartition then most of the fields of a thread Id can be copied from 
		// its parent LPU configuration
		if (lps->isSubpartitionSpace()) {
			functionBody << indent << varName << ".groupId = 0" << stmtSeparator;	
			functionBody << indent << varName << ".ppuCount = 1" << stmtSeparator;
			functionBody << indent << varName << ".groupSize = ";
			functionBody << namePrefix << parentLps->getName() << "].groupSize";
			functionBody << stmtSeparator;
			functionBody << indent << varName << ".id = 0" << stmtSeparator;
			functionBody << indent;
			functionBody << "idsArray[Space_" << lps->getName() << "] = idsArray[Space_";
			functionBody << parentLps->getName() << "]" << stmtSeparator << std::endl;
			continue;
		}

		// determine the total number of threads contributing in the parent PPS and current thread's 
		// index in that PPS 
		if (parent == mappingRoot) {
			functionBody << indent << "threadCount = Max_Total_Threads";
			functionBody << stmtSeparator;
			groupThreadIdStr << "idsArray[Space_Root]";
		} else {
			functionBody << indent;
			functionBody << "threadCount = " << namePrefix << parentLps->getName() << "].groupSize";
			functionBody << stmtSeparator;
			groupThreadIdStr << "idsArray[Space_" << parentLps->getName() << "]";
		}

		// determine the number of threads per group in the current PPS
		functionBody << indent;
		if (lps->getDimensionCount() > 0) {
			functionBody << "groupSize = threadCount" << " / " << partitionCount;
		} else 	functionBody << "groupSize = threadCount";
		functionBody << stmtSeparator;

		// determine the id of the thread in the group it belongs to	
		functionBody << indent;
		functionBody << "groupThreadId = " << groupThreadIdStr.str() << " \% groupSize";
		functionBody << stmtSeparator;

		// assign proper group Id, PPU count, and group size in the PPU-Ids variable created before 
		functionBody << indent;
		functionBody  << varName << ".groupId = " << groupThreadIdStr.str() << " / groupSize";
		functionBody << stmtSeparator;	
		functionBody << indent;
		functionBody  << varName << ".ppuCount = " << partitionCount;
		functionBody << stmtSeparator;
		functionBody << indent;
		functionBody  << varName << ".groupSize = groupSize";
		functionBody << stmtSeparator;

		// assign PPU Id to the thread depending on its groupThreadId
		functionBody << indent;
		functionBody << "if (groupThreadId == 0) " << varName << ".id\n"; 
		functionBody << tripleIndent <<  "= " << varName << ".groupId";
		functionBody << stmtSeparator;	
		functionBody << indent;
		functionBody << "else " << varName << ".id = INVALID_ID";
		functionBody << stmtSeparator;	
		
		// store the index of the thread in the group for subsequent references	
		functionBody << indent;
		functionBody << "idsArray[Space_" << lps->getName() << "] = groupThreadId";
		functionBody << stmtSeparator;
		functionBody << std::endl;
	}
	functionBody << indent << "return threadIds" << stmtSeparator;
	functionBody << "}\n";

	headerFile << "ThreadIds *" << functionHeader.str() << ";\n\n";	
	programFile << std::endl << "ThreadIds *" << initials << "::"; 
	programFile <<functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

void generateFnForThreadIdsAdjustment(const char *headerFileName, 
                const char *programFileName, const char *initials, MappingNode *mappingRoot) {

	std::ofstream programFile, headerFile;
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open header/program file";
		std::exit(EXIT_FAILURE);
	}
                
	std::ostringstream functionHeader;
        functionHeader << "adjustPpuCountsAndGroupSizes(ThreadIds *threadId)";
        std::ostringstream functionBody;
        
	functionBody << " {\n\n";

	// declare two local variables to keep track of the thread index ranges of current thread's group as the flow of
	// control moves downward from upper to lower LPSes
	functionBody << indent << "int groupBegin = 0" << stmtSeparator;
	functionBody << indent << "int groupEnd = Total_Threads - 1" << stmtSeparator;
	functionBody << '\n';

	// declare some other local variables to temporarily hold group Ids and counts
	functionBody << indent << "int groupId = 0" << stmtSeparator;
	functionBody << indent << "int groupSize = Total_Threads" << stmtSeparator;
	functionBody << indent << "int ppuCount = 1" << stmtSeparator;
	functionBody << '\n';

	// iterate over the mapping nodes in FCFS order	
	std::deque<MappingNode*> nodeQueue;
        for (int i = 0; i < mappingRoot->children->NumElements(); i++) {
        	nodeQueue.push_back(mappingRoot->children->Nth(i));
        }
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		Space *lps = node->mappingConfig->LPS;
		const char *lpsName = lps->getName();
		
		// create a prefix and variable name to make future references easy
		std::string namePrefix = "threadId->ppuIds[Space_";
		std::ostringstream varNameStr;
		varNameStr << namePrefix << lps->getName() << "]";
		std::string varName = varNameStr.str();

		// if the LPS is a subpartition space then the PPU count is default 1 and we can assign the group size
		// of its parent to its group size
		if (lps->isSubpartitionSpace()) {
			functionBody << indent << varName << ".groupSize = groupEnd - groupBegin + 1";
			functionBody << stmtSeparator << '\n';
			continue;
		}

		// otherwise we need to do some actual count and assumed count comparison; first retrieves the values 
		// of some interesting variable
		functionBody << indent << "groupId = " << varName << ".groupId" << stmtSeparator;
		functionBody << indent << "groupSize = " << varName << ".groupSize" << stmtSeparator;

		// determine the PPU count at the current level
		functionBody << indent << "ppuCount = ((groupEnd - groupBegin + 1) + (groupSize - 1)) / groupSize";
		functionBody << stmtSeparator;
		functionBody << indent << varName << ".ppuCount = ppuCount" << stmtSeparator;

		// determine the thread Id range to be partitioned by next level
		functionBody << indent << "groupBegin = groupId * groupSize" << stmtSeparator;
		functionBody << indent << "groupEnd = min(groupBegin + groupSize - 1" << paramSeparator;
		functionBody << "groupEnd)" << stmtSeparator;

		// determine the group size as the number of IDs within the updated range
		functionBody << indent << varName << ".groupSize = groupEnd - groupBegin + 1" << stmtSeparator;

		functionBody << '\n';
	}
	 
	functionBody << "}\n";
	
	headerFile << "void " << functionHeader.str() << ";\n\n";	
	programFile << std::endl << "void " << initials << "::"; 
	programFile <<functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;

	headerFile.close();
	programFile.close();
}

void generateLpuDataStructures(const char *outputFile, 
		MappingNode *mappingRoot, List<ReductionMetadata*> *reductionInfos) {
       
	std::cout << "Generating data structures for LPUs\n";
 
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
		const char *message = "data structures representing LPUs";
		decorator::writeSectionHeader(programFile, message);
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

		std::ostringstream header;
		header << "Space " << lps->getName();
		decorator::writeSubsectionHeader(programFile, header.str().c_str());

		// create the object for representing an LPU of the LPS
		programFile << "class Space" << lps->getName() << "_LPU : public LPU {\n";
		programFile << "  public:\n";
		for (int i = 0; i < localArrays->NumElements(); i++) {
			ArrayDataStructure *array = (ArrayDataStructure*) lps->getLocalStructure(localArrays->Nth(i));
			ArrayType *arrayType = (ArrayType*) array->getType();
			const char *elemType = arrayType->getTerminalElementType()->getName();
			programFile << indent << elemType << " *" << array->getName();
			programFile << stmtSeparator;
			
			// if there are multiple epoch version needed for the array in current LPS then create references
			// for all previous epoch versions
			int versionCount = array->getLocalVersionCount();
			for (int j = 1; j <= versionCount; j++) {
				programFile << indent << elemType << " *" << array->getName();
				programFile << "_lag_" << j;
				programFile << stmtSeparator;
			}
			
			int dimensions = array->getDimensionality();
			programFile << indent << "PartDimension ";
			programFile << array->getName() << "PartDims[" << array->getDimensionality() << "]";
			programFile << stmtSeparator;	
		}

		// add a specific lpu_id static array with dimensionality equals to the dimensions of the LPS
		if (lps->getDimensionCount() > 0) {
			programFile << indent << "int lpuId[";
			programFile << lps->getDimensionCount() << "]";
			programFile << stmtSeparator;
		}

		// define a print function for the LPU
		programFile << std::endl;
		programFile << indent << "void print(std::ofstream &stream, int indent)" << stmtSeparator;	
		programFile << "};\n";
	}
	
	programFile.close();
}

void generatePrintFnForLpuDataStructures(const char *initials, 
		const char *outputFile, 
		MappingNode *mappingRoot, List<ReductionMetadata*> *reductionInfos) {

	std::cout << "Generating print functions for LPUs\n";
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
		const char *message = "LPU print functions";
		decorator::writeSectionHeader(programFile, message);
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
			programFile << indent << "for (int i = 0; i < indentLevel; i++) ";
			programFile << "stream << '\\t'" << stmtSeparator;
			programFile << indent << "stream << \"Array: " << arrayName << "\"";
			programFile << " << std::endl";
			programFile << stmtSeparator;
			for (int j = 0; j < dimensions; j++) {
				programFile << indent;
				programFile << arrayName << "PartDims[" << j;
				programFile  << "].print(stream, indentLevel + 1)";
				programFile << stmtSeparator;
			}
		}

		programFile << indent << "stream.flush()" << stmtSeparator;
		programFile << "}\n";
	}

	programFile << std::endl;
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
		const char *header = "Data structures for Array-Metadata and Environment-Links";
		decorator::writeSectionHeader(programFile, header);	
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
	programFile << "static ArrayMetadata arrayMetadata" << statementSeparator;
	
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
	programFile << "static EnvironmentLinks environmentLinks" << statementSeparator << std::endl;
	programFile.close();
	return linkList;
}

void generateFnForMetadataAndEnvLinks(const char *taskName, const char *initials, 
		const char *outputFile, MappingNode *mappingRoot,
                List<const char*> *externalLinks) {

	std::cout << "Generating function implementations for array metadata and environment links\n";
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
		const char *header = "Functions for ArrayMetadata and EnvironmentLinks";
		decorator::writeSectionHeader(programFile, header);
	} else {
		std::cout << "Unable to open output program file";
		std::exit(EXIT_FAILURE);
	}
	
	Space *rootLps = mappingRoot->mappingConfig->LPS;

	// generate constructor for array metadata 
	programFile << std::endl << initials << "::ArrayMetadata::ArrayMetadata() : Metadata() {\n";
	programFile << indent << "setTaskName";
	programFile << "(\"" << taskName << "\")" << stmtSeparator;  
	programFile << "}" << std::endl << std::endl;

	// generate a print function for array metadata
	programFile << "void " << initials << "::ArrayMetadata::" << "print(std::ofstream &stream) {\n";
	programFile << indent << "stream << \"Array Metadata\" << std::endl" << stmtSeparator;
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		const char *arrayName = localArrays->Nth(i);
		programFile << indent << "stream << \"Array: " << arrayName << "\"";
		programFile << stmtSeparator;
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(arrayName);
		int dimensions = array->getDimensionality();
		for (int j = 0; j < dimensions; j++) {
			programFile << indent << "stream << ' '" << stmtSeparator;
			programFile << indent << arrayName << "Dims[" << j << "].print(stream)";
			programFile << stmtSeparator;
		}
		programFile << indent << "stream << std::endl" << stmtSeparator;
	}
	programFile << indent << "stream.flush()" << stmtSeparator << "}\n";
	
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

	// include the PartDimension class from compiler library to store metadata information for environment
	// references
	headerFile << "#include \"../../../common-libs/domain-obj/structure.h\"\n\n"; 
	
	// first have a list of forward declarations for all tuples to avoid having errors during 
	// compilation of individual classes
	for (int i = 0; i < tupleDefList->NumElements(); i++) {
		TupleDef *tupleDef = tupleDefList->Nth(i);
		headerFile << "class " << tupleDef->getId()->getName() << ";\n";
	}
	headerFile << "\n";

	// then generate a class for each tuple in the list
	for (int i = 0; i < tupleDefList->NumElements(); i++) {
		// retrieve the tuple definition
		TupleDef *tupleDef = (TupleDef*) tupleDefList->Nth(i);
		List<VariableDef*> *variables = tupleDef->getComponents();
		// generate a new class and add the elements as public components
		headerFile << "class " << tupleDef->getId()->getName();
		if (tupleDef->isEnvironment()) {
			headerFile << " : public EnvironmentBase";
		} 
		headerFile << " {\n";
		headerFile << "  public:\n";
		for (int j = 0; j < variables->NumElements(); j++) {
			headerFile << "\t";
			VariableDef *variable = variables->Nth(j);
			Type *type = variable->getType();
			const char *varName = variable->getId()->getName();
			headerFile << type->getCppDeclaration(varName);
			headerFile << ";\n";
                      	// include a metadata property in the class if the current property is a dynamic array
                       	ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
                       	StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                       	if (arrayType != NULL && staticArray == NULL) {
                               	int dimensions = arrayType->getDimensions();
                               	headerFile << "\tPartDimension ";
                        	headerFile << varName << "Dims[" << dimensions << "];\n";
                       	}

		}
		// create a constructor for the class
		headerFile << "\t" << tupleDef->getId()->getName() << "()";
		if (tupleDef->isEnvironment()) {
			headerFile << " : EnvironmentBase()";
		} 
		headerFile << " {\n";
		for (int j = 0; j < variables->NumElements(); j++) {
			VariableDef *variable = variables->Nth(j);
			Type *type = variable->getType();
			const char *varName = variable->getId()->getName();
			if (type == Type::boolType) {
				headerFile << "\t\t" << varName << " = false;\n";
			} else if (type == Type::intType 
					|| type == Type::floatType 
					|| type == Type::doubleType
					|| type == Type::charType) {
				headerFile << "\t\t" << varName << " = 0;\n";
			} else if (type == Type::stringType || dynamic_cast<ArrayType*>(type) != NULL) {
				headerFile << "\t\t" << varName << " = NULL;\n";
			}
		}	
		headerFile << "\t}\n";
		headerFile << "};\n\n";
	}

	headerFile << "#endif\n";
	headerFile.close();
}

void generateClassesForGlobalScalars(const char *filePath, List<TaskGlobalScalar*> *globalList, Space *rootLps) {
	
	std::cout << "Generating structures holding task global and thread local scalar\n";

	std::ofstream headerFile;
	headerFile.open (filePath, std::ofstream::out | std::ofstream::app);
	if (!headerFile.is_open()) {
		std::cout << "Unable to open output header file for task\n";
		std::exit(EXIT_FAILURE);
	}
                
        const char *message = "Data structures for Task-Global and Thread-Local scalar variables";
	decorator::writeSectionHeader(headerFile, message);	
	headerFile << std::endl;
	
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
		const char *varName = scalar->getName();
		*stream << indent;
		*stream << type->getCppDeclaration(varName);
		*stream << stmtSeparator;
		
		// check if there are multiple versions for the variable; if YES then create copies for
		// other time-lagged versions
		DataStructure *structure = rootLps->getStructure(varName);
		int versionCount = structure->getVersionCount();
		for (int j = 1; j <= versionCount; j++) {
			std::ostringstream oldVersionName;
			oldVersionName << varName << "_lag_" << j;
			*stream << indent;
			*stream << type->getCppDeclaration(oldVersionName.str().c_str());
			*stream << stmtSeparator;
		}
	}
	
	taskGlobals << "};\n\n";
	threadLocals << "};\n";

	headerFile << taskGlobals.str() << threadLocals.str();
	headerFile.close();
}

void generateInitializeFunction(const char *headerFileName, const char *programFileName, const char *initials,
                List<const char*> *envLinkList, TaskDef *taskDef, Space *rootLps) {

        std::cout << "Generating function for the initialize block\n";

        std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file for initialize block generation";
                std::exit(EXIT_FAILURE);
        }

        const char *header = "function for the initialize block";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);

        // put five default parameters for metadata, env-Links, task-globals, thread-locals, and partition 
        // configuration
        std::ostringstream functionHeader;
        functionHeader << "initializeTask(ArrayMetadata *arrayMetadata";
        functionHeader << paramSeparator << '\n' << doubleIndent;
        functionHeader << "EnvironmentLinks environmentLinks";
        functionHeader << paramSeparator << '\n' << doubleIndent;
        functionHeader << "TaskGlobals *taskGlobals";
        functionHeader << paramSeparator << '\n' << doubleIndent;
        functionHeader << "ThreadLocals *threadLocals";
        functionHeader << paramSeparator << '\n' << doubleIndent;
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
                                functionBody << indent;
                                functionBody << varName << "[" << j << "]";
                                functionBody << " = " << "environmentLinks.";
                                functionBody << envLink;
                                functionBody << "Dims[" << j << "]";
                                functionBody << stmtSeparator;
                        }
                // otherwise the value of the scalar variable should be copied back to task global or thread
                // local variable depending on what is the right destination
                } else {
                        functionBody << indent;
                        functionBody << transformer->getTransformedName(envLink, true, false);
                        functionBody << " = " << "environmentLinks.";
                        functionBody << envLink;
                        functionBody << stmtSeparator;
                }
        }

        InitializeSection *initSection = taskDef->getInitSection();
        if (initSection != NULL) {

                // iterate over all initialization parameters and add them as function arguments
                List<const char*> *argNames = initSection->getArguments();
                List<Type*> *argTypes = initSection->getArgumentTypes();
                for (int i = 0; i < argNames->NumElements(); i++) {
                        const char *arg = argNames->Nth(i);
                        Type *type = argTypes->Nth(i);
                        functionHeader << paramSeparator;
                        functionHeader << "\n" << doubleIndent;
                        functionHeader << type->getCppDeclaration(arg);
                        // if any argument matches a global variable in the task then copy it to the appropriate
                        // data structure
                        if (transformer->isThreadLocal(arg) || transformer->isTaskGlobal(arg)) {
                                functionBody << indent;
                                functionBody << transformer->getTransformedName(arg, true, false);
                                functionBody << " = " << arg;
                                functionBody << stmtSeparator;
                        }
                }

                // then translate the user code in the init section into a c++ instruction stream
                initSection->generateCode(functionBody);
        }

        functionHeader << ")";
        functionBody << "}\n";

        headerFile << std::endl << "void " << functionHeader.str() << ";\n\n";
        programFile << std::endl << "void " << initials << "::";
        programFile <<functionHeader.str() << " " << functionBody.str();
        programFile << std::endl;

        headerFile.close();
        programFile.close();
}


// an auxiliary function to be used by the function immediately following it to group extern library links
void groupLibrayLinkInfo(Hashtable<List<const char*>*> *languageLibraryMap, 
		List<const char*> *languageList, 
		IncludesAndLinksMap *externConfig) {

	List<const char*> *languages = externConfig->getLanguagesUsed();
	for (int j = 0; j < languages->NumElements(); j++) {
		const char *language = languages->Nth(j);
		List<const char*> *libraries = NULL;
		if (string_utils::contains(languageList, language)) {
			libraries = languageLibraryMap->Lookup(language);
		} else {
			libraries = new List<const char*>;
			languageLibraryMap->Enter(language, libraries);
			languageList->Append(language);
		}
		LanguageIncludesAndLinks *languageConfig
				= externConfig->getIncludesAndLinksForLanguage(language);
		string_utils::combineLists(libraries, languageConfig->getLibraryLinks());
	}
}

void generateExternLibraryLinkInfo(const char *linkDescriptionFile) {
        
        List<const char*> *languageList = new List<const char*>;
        Hashtable<List<const char*>*> *languageLibraryMap = new Hashtable<List<const char*>*>;

        // get all tasks and group their library linkage annotations from different extern code block by 
	// language
	List<Definition*> *taskDefs = ProgramDef::program->getComponentsByType(TASK_DEF);
        for (int i = 0; i < taskDefs->NumElements(); i++) {
                TaskDef *taskDef = (TaskDef*) taskDefs->Nth(i);
                IncludesAndLinksMap *externConfig = taskDef->getExternBlocksHeadersAndLibraries();
        	groupLibrayLinkInfo(languageLibraryMap, languageList, externConfig);
	}

	// do the same thing for all functions
	List<Definition*> *fnDefs = ProgramDef::program->getComponentsByType(FN_DEF);
        for (int i = 0; i < fnDefs->NumElements(); i++) {
                FunctionDef *fnDef = (FunctionDef*) fnDefs->Nth(i);
                IncludesAndLinksMap *externConfig = fnDef->getExternBlocksHeadersAndLibraries();
        	groupLibrayLinkInfo(languageLibraryMap, languageList, externConfig);
	}

        // write the libraries to be included by their language type on the library description file
        std::ofstream descriptionFile;
        descriptionFile.open (linkDescriptionFile, std::ofstream::out);
        if (!descriptionFile.is_open()) {
                std::cout << "Unable to open extern block linkage description file";
                std::exit(EXIT_FAILURE);
        }
        for (int i = 0; i < languageList->NumElements(); i++) {
                const char *language = languageList->Nth(i);
                descriptionFile << language << " =";
                List<const char*> *libraryLinks = languageLibraryMap->Lookup(language);
                for (int j = 0; j < libraryLinks->NumElements(); j++) {
                        descriptionFile << ' '  << '-' << libraryLinks->Nth(j);
                }
                descriptionFile << '\n';
        }
        descriptionFile.close();
}
