#include "communication.h"
#include "space_mapping.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../utils/common_utils.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
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
#include <queue>


void generateDistributionTreeFnForStructure(const char *varName,
                std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, Space *rootLps) {

	decorator::writeSubsectionHeader(headerFile, varName);
	decorator::writeSubsectionHeader(programFile, varName);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;

	fnHeader << "generateDistributionTreeFor_" << varName << "(";
	fnHeader << "List<SegmentState*> *segmentList" << paramSeparator << '\n';
	fnHeader << doubleIndent << "Hashtable<DataPartitionConfig*> *configMap)";

	// find the list of LPSes that allocate the variable; the distribution tree should have branches for all those LPSes
	List<Space*> *releventLpses = new List<Space*>;
	std::deque<Space*> spaceQueue;
	spaceQueue.push_back(rootLps);
	while (!spaceQueue.empty()) {
		Space *currentLps = spaceQueue.front();
		spaceQueue.pop_front();
		List<Space*> *children = currentLps->getChildrenSpaces();
		for (int i = 0; i < children->NumElements(); i++) {
			spaceQueue.push_back(children->Nth(i));
		}
		if (currentLps->getSubpartition() != NULL) {
			spaceQueue.push_back(currentLps->getSubpartition());
		}
		if (currentLps->allocateStructure(varName)) {
			releventLpses->Append(currentLps);
		}
	}

	fnBody << "{\n\n";
	
	// first create the root of the distribution tree
	fnBody << indent;
	fnBody<< "BranchingContainer *rootContainer = new BranchingContainer(0, LpsDimConfig())";
	fnBody << stmtSeparator;

	// iterate over all segments
	fnBody << indent << "for (int i = 0; i < segmentList->NumElements(); i++) {\n";
	fnBody << doubleIndent << "SegmentState *segment = segmentList->Nth(i)" << stmtSeparator;
	fnBody << doubleIndent << "int segmentTag = segment->getPhysicalId()" << stmtSeparator;
	fnBody << doubleIndent << "List<ThreadState*> *threadList = segment->getParticipantList()" << stmtSeparator;

	// iterate over the threads of the current segments
	fnBody << doubleIndent << "for (int j = 0; j < threadList->NumElements(); j++) {\n";
	fnBody << tripleIndent << "ThreadState *thread = threadList->Nth(j)" << stmtSeparator;

	// iterator over the list of LPSes that allocate this structures
	fnBody << tripleIndent << "int lpuId = INVALID_ID" << stmtSeparator;
	fnBody << tripleIndent << "DataPartitionConfig *partConfig = NULL" << stmtSeparator;
	fnBody << tripleIndent << "List<int*> *partId = NULL" << stmtSeparator;
	fnBody << tripleIndent << "DataItemConfig *dataItemConfig = NULL" << stmtSeparator;
	fnBody << tripleIndent << "std::vector<LpsDimConfig> *dimOrder = NULL" << stmtSeparator;
	for (int i = 0; i < releventLpses->NumElements(); i++) {
		Space *lps = releventLpses->Nth(i);
		const char *lpsName = lps->getName();
		fnBody << "\n";
		fnBody << tripleIndent << "//generating parts for: " << lpsName << "\n";
		fnBody << tripleIndent << "partConfig = configMap->Lookup(\"";
		fnBody << varName << "Space" << lpsName << "Config" << "\")" << stmtSeparator;
		fnBody << tripleIndent << "partId = partConfig->generatePartIdTemplate()" << stmtSeparator;
		fnBody << tripleIndent << "dataItemConfig = partConfig->generateStateFulVersion()";
		fnBody << stmtSeparator << tripleIndent;
		fnBody << "dimOrder = dataItemConfig->generateDimOrderVector()" << stmtSeparator;	
		
		// generate the LPU Ids for the current LPS and Thread combination and retrieve data part IDs from LPU IDs
		fnBody << tripleIndent << "while((lpuId = thread->getNextLpuId(";
		fnBody << "Space_" << lpsName << paramSeparator;
		fnBody << "Space_" << rootLps->getName() << paramSeparator;
		fnBody << "lpuId)) != INVALID_ID) {\n";
		fnBody << quadIndent << "List<int*> *lpuIdChain = thread->getLpuIdChainWithoutCopy(";
		fnBody << std::endl << quadIndent << doubleIndent;
		fnBody << "Space_" << lpsName << paramSeparator;
		fnBody << "Space_" << rootLps->getName() << ")" << stmtSeparator;
		fnBody << quadIndent << "partConfig->generatePartId(lpuIdChain" << paramSeparator;
		fnBody << "partId)"  << stmtSeparator;
		fnBody << quadIndent << "rootContainer->insertPart(*dimOrder" << paramSeparator;
		fnBody << "segmentTag" << paramSeparator << "partId)" << stmtSeparator;
		fnBody << tripleIndent << "}\n";
	}	

	fnBody << doubleIndent << "}\n";		
	fnBody << indent << "}\n";

	fnBody << indent << "return rootContainer" << stmtSeparator;
	fnBody << "}\n";

	headerFile << "Container *" << fnHeader.str() << ";\n";
	programFile << "\nContainer *" << initials << "::" << fnHeader.str() << " " << fnBody.str();
}

void generateFnsForDistributionTrees(const char *headerFileName,
                const char *programFileName,
                TaskDef *taskDef,
                List<PPS_Definition*> *pcubesConfig) {

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for generating distribution tree functions";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file for generating distribution tree functions";
                std::exit(EXIT_FAILURE);
        }

	const char *initials = string_utils::getInitials(taskDef->getName());
        initials = string_utils::toLower(initials);

	// determine the id of the PPS where memory segmentation takes place; this is needed to determine if communication is
	// needed for synchronizing shared variables
	int segmentedPPS = pcubesConfig->NumElements();
	for (int i = 0; i < pcubesConfig->NumElements(); i++) {
		PPS_Definition *pps = pcubesConfig->Nth(i);
		if (pps->segmented) {
			segmentedPPS = pps->id;
			break;
		}
	}

	// get the list of arrays to be involved in some form of communication; for scalar variables no distribution tree is
	// needed
	List<const char*> *syncVars = taskDef->getComputation()->getVariablesNeedingCommunication(segmentedPPS);
	List<const char*> *syncArrays = new List<const char*>;
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	for (int i = 0; i < syncVars->NumElements(); i++) {
		const char *varName = syncVars->Nth(i);
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) syncArrays->Append(varName);
	}

	if (syncArrays->NumElements() > 0) {

		std::cout << "Generating part distribution trees for communicated variables\n";

		const char *message = "functions to generate distribution trees for communicated variables";
		decorator::writeSectionHeader(headerFile, message);
		decorator::writeSectionHeader(programFile, message);
		for (int i = 0; i < syncArrays->NumElements(); i++) {
			const char *varName = syncArrays->Nth(i);
			generateDistributionTreeFnForStructure(varName, headerFile, programFile, initials, rootLps);
		}
		generateFnForDistributionMap(headerFile, programFile, initials, syncArrays);
	}

	headerFile.close();
	programFile.close();
}

void generateFnForDistributionMap(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, List<const char*> *varList) {

	const char *message = "distribution map";
	decorator::writeSubsectionHeader(headerFile, message);
	decorator::writeSubsectionHeader(programFile, message);

	std::ostringstream fnHeader, fnBody;
	fnHeader << "generateDistributionMap(";
	fnHeader << "List<SegmentState*> *segmentList" << paramSeparator << '\n';
	fnHeader << doubleIndent << "Hashtable<DataPartitionConfig*> *configMap)";
	
	fnBody << "{\n\n";
	fnBody << indent << "PartDistributionMap *distributionMap = new PartDistributionMap()" << stmtSeparator;
	for (int i = 0 ; i < varList->NumElements(); i++) {
		const char *varName = varList->Nth(i);
		fnBody << indent << "distributionMap->setDistributionForVariable(\"";
		fnBody << varName << "\"" << paramSeparator << '\n';
		fnBody << indent << doubleIndent;
		fnBody << "generateDistributionTreeFor_" << varName << "(segmentList" << paramSeparator;
		fnBody << "configMap))" << stmtSeparator;
	}	
	fnBody << indent << "return distributionMap" << stmtSeparator;
	fnBody << "}\n";

	headerFile << "PartDistributionMap *" << fnHeader.str() << ";\n";
	programFile << "\nPartDistributionMap *" << initials << "::" << fnHeader.str() << " " << fnBody.str();
}
