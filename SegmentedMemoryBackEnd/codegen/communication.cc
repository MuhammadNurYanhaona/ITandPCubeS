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
#include "../static-analysis/sync_stat.h"

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

		std::cout << "\tGenerating part distribution trees for communicated variables\n";

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

void generateConfinementConstrConfigFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, CommunicationCharacteristics *commCharacter) {

	const char *dependencyName = commCharacter->getSyncRequirement()->getDependencyArc()->getArcName();
	
	decorator::writeSubsectionHeader(headerFile, dependencyName);
	decorator::writeSubsectionHeader(programFile, dependencyName);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;

	fnHeader << "getConfineConstrConfigFor_" << dependencyName << "(TaskData *taskData" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "int localSegmentTag" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "PartDistributionMap *distributionMap)";

	const char *varName = commCharacter->getVarName();
	const char *senderSyncLpsName = commCharacter->getSenderSyncSpace()->getName();
	const char *senderAllocatorLpsName = commCharacter->getSenderDataAllocatorSpace()->getName();
	const char *receiverSyncLpsName = commCharacter->getReceiverSyncSpace()->getName();
	const char *receiverAllocatorLpsName = commCharacter->getReceiverDataAllocatorSpace()->getName();
	
	fnBody << "{\n\n";

	// generate local variables for all confinement construction properties
	fnBody << indent << "int senderLps = Space_" << senderSyncLpsName << stmtSeparator;
	fnBody << indent << "DataItemConfig *senderDataConfig = partConfigMap->Lookup(";
	fnBody << '\n' << indent << doubleIndent << "\"";
	fnBody << varName << "Space" << senderAllocatorLpsName << "Config"<< "\")->";
	fnBody << "generateStateFulVersion()" << stmtSeparator;
	fnBody << indent << "int receiverLps = Space_" << receiverSyncLpsName << stmtSeparator; 
	fnBody << indent << "DataItemConfig *receiverDataConfig = partConfigMap->Lookup(";
	fnBody << '\n' << indent << doubleIndent << "\"";
	fnBody << varName << "Space" << senderAllocatorLpsName << "Config"<< "\")->";
	fnBody << "generateStateFulVersion()" << stmtSeparator;
	fnBody << indent << "int confinementLps = Space_" << commCharacter->getConfinementSpace()->getName();
	fnBody << stmtSeparator << indent;
	fnBody << "PartIdContainer *senderPartTree = NULL" << stmtSeparator;
	fnBody << indent <<  "DataItems *senderDataItems = taskData->getDataItemsOfLps(\"Space";
	fnBody << senderAllocatorLpsName << "\"" << paramSeparator;
	fnBody << "\"" << varName << "\")" << stmtSeparator;
	fnBody << indent << "if(senderDataItems != NULL) senderPartTree = ";
	fnBody << "senderDataItems->getPartIdContainer()" << stmtSeparator;
	fnBody << indent << "PartIdContainer *receiverPartTree = NULL" << stmtSeparator;
	fnBody << indent << "DataItems *receiverDataItems = taskData->getDataItemsOfLps(\"Space";
	fnBody << receiverAllocatorLpsName << "\"" << paramSeparator;
	fnBody << "\"" << varName << "\")" << stmtSeparator;
	fnBody << indent << "if(receiverDataItems != NULL) receiverPartTree = ";
	fnBody << "receiverDataItems->getPartIdContainer()" << stmtSeparator;
	fnBody << indent << "BranchingContainer *distributionTree = ";
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "(BranchingContainer*) distributionMap->getDistrubutionTree(\"";
	fnBody << varName << "\")" << stmtSeparator;

	// create a new instance of the confinement construction config
	fnBody << '\n' << indent << "ConfinementConstructionConfig *confinementConfig = ";
	fnBody << "new ConfinementConstructionConfig(localSegmentTag" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "senderLps" << paramSeparator << "senderDataConfig" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "receiverLps" << paramSeparator << "receiverDataConfig" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "confinementLps" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "senderPartTree" << paramSeparator << "receiverPartTree" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "distributionTree)" << stmtSeparator;

	fnBody << indent << "return confinementConfig" << stmtSeparator;
	fnBody << "}\n";

	headerFile << "ConfinementConstructionConfig *" << fnHeader.str() << stmtSeparator;
	programFile << "\nConfinementConstructionConfig *" << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();
}

List<CommunicationCharacteristics*> *generateFnsForConfinementConstrConfigs(const char *headerFileName,
                const char *programFileName, 
		TaskDef *taskDef, List<PPS_Definition*> *pcubesConfig) {

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for generating confinement configuration functions";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file for generating confinement configuration functions";
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
	
	// select the communication requirements that involves some arrays; for scalar variables no confinement construction
	// configuration is needed as their logic of synchronization does not involve confinements 
	List<CommunicationCharacteristics*> *commCharacterList 
			= taskDef->getComputation()->getCommCharacteristicsForSyncReqs(segmentedPPS);
	List<CommunicationCharacteristics*> *arrayComms = new List<CommunicationCharacteristics*>;
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		CommunicationCharacteristics *currComm = commCharacterList->Nth(i);
		const char *varName = currComm->getVarName();	
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) arrayComms->Append(currComm);
	}
	
	if (arrayComms->NumElements() > 0) {

		std::cout << "\tGenerating functions for communication confinement configurations\n";

		const char *message = "functions to generate communication confinement configurations";
		decorator::writeSectionHeader(headerFile, message);
		decorator::writeSectionHeader(programFile, message);
		for (int i = 0; i < arrayComms->NumElements(); i++) {
			CommunicationCharacteristics *commCharacter = arrayComms->Nth(i);
			generateConfinementConstrConfigFn(headerFile,programFile, initials, commCharacter);
		}
	}

	headerFile.close();
	programFile.close();

	// return the list of communication characteristics to be used by later code generation functions
	return commCharacterList;
}

void generateFnForDataExchanges(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, 
		Space *rootSpace, CommunicationCharacteristics *commCharacter) {
	
	const char *dependencyName = commCharacter->getSyncRequirement()->getDependencyArc()->getArcName();
	
	decorator::writeSubsectionHeader(headerFile, dependencyName);
	decorator::writeSubsectionHeader(programFile, dependencyName);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;
	
	fnHeader << "getDataExchangeListFor_" << dependencyName << "(TaskData *taskData" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "int localSegmentTag" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "PartDistributionMap *distributionMap)";

	fnBody << "{\n\n";

	// first instanciate a confinment construction configuration for the dependency arc by calling the designated function
	// for the current dependency
	fnBody << indent << "ConfinementConstructionConfig *ccConfig = ";
	fnBody << "getConfineConstrConfigFor_" << dependencyName << "(taskData" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "partConfigMap" << paramSeparator;
	fnBody << "localSegmentTag" << paramSeparator;
	fnBody << "distributionMap)" << stmtSeparator;

	// then generate the list of confinements relevent to the current segment for the concerned data dependency
	fnBody << indent << "List<Confinement*> *confinementList = ";
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "Confinement::generateAllConfinements(ccConfig" << paramSeparator;
	fnBody << "Space_" << rootSpace->getName() << ')' << stmtSeparator;

	// if there is no confinement in the list then there is no data to exchange
	fnBody << indent << "if (confinementList == NULL || confinementList->NumElements() == 0) return NULL";
	fnBody << stmtSeparator << '\n';

	// instanciate a new list for data exchanges and pick up exchanges in it by traversing individual confinements
	fnBody << indent << "List<DataExchange*> *dataExchangeList = new List<DataExchange*>" << stmtSeparator;
	fnBody << indent << "for (int i = 0; i < confinementList->NumElements(); i++) {\n";
	fnBody << doubleIndent << "Confinement *confinement = confinementList->Nth(i)" << stmtSeparator;
	fnBody << doubleIndent << "List<DataExchange*> *confinementExchanges = confinement->getAllDataExchanges()";
	fnBody << stmtSeparator;
	fnBody << doubleIndent << "if (confinementExchanges != NULL) {\n";
	fnBody << tripleIndent << "dataExchangeList->AppendAll(confinementExchanges)" << stmtSeparator;
	fnBody << tripleIndent << "delete confinementExchanges" << stmtSeparator;
	fnBody << doubleIndent << "}\n";
	fnBody << indent << "}\n";

	// if there is no exchange in the list then return NULL otherwise return the list
	fnBody << indent << "if (dataExchangeList->NumElements() == 0) {\n";
	fnBody << doubleIndent << "delete dataExchangeList" << stmtSeparator;
	fnBody << doubleIndent << "return NULL" << stmtSeparator;
	fnBody << indent << "}\n";
	fnBody << indent << "return dataExchangeList" << stmtSeparator;

	fnBody << "}\n"; 	
	
	headerFile << "List<DataExchange*> *" << fnHeader.str() << stmtSeparator;
	programFile << "\nList<DataExchange*> *" << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();
}

void generateAllDataExchangeFns(const char *headerFileName,
                const char *programFileName,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList) {

	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for generating data-exchange functions";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout << "Unable to open output header file for generating data-exchange functions";
                std::exit(EXIT_FAILURE);
        }

	const char *initials = string_utils::getInitials(taskDef->getName());
        initials = string_utils::toLower(initials);
	
	List<CommunicationCharacteristics*> *arrayComms = new List<CommunicationCharacteristics*>;
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		CommunicationCharacteristics *currComm = commCharacterList->Nth(i);
		const char *varName = currComm->getVarName();	
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) arrayComms->Append(currComm);
	}
	
	if (arrayComms->NumElements() > 0) {

		std::cout << "\tGenerating functions for data-exchange list generation\n";
		
		const char *message = "functions to generate data exchange lists for data dependencies";
		decorator::writeSectionHeader(headerFile, message);
		decorator::writeSectionHeader(programFile, message);
		for (int i = 0; i < arrayComms->NumElements(); i++) {
			CommunicationCharacteristics *commCharacter = arrayComms->Nth(i);
			generateFnForDataExchanges(headerFile, programFile, initials, rootLps, commCharacter);
		}
	}

	headerFile.close();
	programFile.close();
}
