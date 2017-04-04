#include "communication.h"
#include "space_mapping.h"
#include "code_constant.h"
#include "task_global.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"
#include "../../../../common-libs/utils/common_utils.h"
#include "../../../../common-libs/utils/decorator_utils.h"

#include "../../../../frontend/src/syntax/ast_def.h"
#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/semantics/task_space.h"
#include "../../../../frontend/src/static-analysis/sync_stat.h"
#include "../../../../frontend/src/codegen-helper/communication_stat.h"

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
	fnBody << "BranchingContainer *rootContainer = new BranchingContainer(0, LpsDimConfig())";
	fnBody << stmtSeparator;
	fnBody << indent << "Assert(rootContainer != NULL)" << stmtSeparator;

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
		fnBody << tripleIndent << "if (thread->isValidPpu(Space_" << lpsName << ")) {\n";
		fnBody << quadIndent << "partConfig = configMap->Lookup(\"";
		fnBody << varName << "Space" << lpsName << "Config" << "\")" << stmtSeparator;
		fnBody << quadIndent << "partId = partConfig->generatePartIdTemplate()" << stmtSeparator;
		fnBody << quadIndent << "dataItemConfig = partConfig->generateStateFulVersion()";
		fnBody << stmtSeparator << quadIndent;
		fnBody << "dimOrder = dataItemConfig->generateDimOrderVector()" << stmtSeparator;	
		
		// generate the LPU Ids for the current LPS and Thread combination and retrieve data part IDs from LPU IDs
		fnBody << std::endl << quadIndent << "while((lpuId = thread->getNextLpuId(";
		fnBody << "Space_" << lpsName << paramSeparator;
		fnBody << std::endl << quadIndent << doubleIndent;
		fnBody << "Space_" << rootLps->getName() << paramSeparator;
		fnBody << "lpuId)) != INVALID_ID) {\n";
		fnBody << quadIndent << indent << "List<int*> *lpuIdChain = thread->getLpuIdChainWithoutCopy(";
		fnBody << std::endl << quadIndent << tripleIndent;
		fnBody << "Space_" << lpsName << paramSeparator;
		fnBody << "Space_" << rootLps->getName() << ")" << stmtSeparator;
		fnBody << quadIndent << indent << "partConfig->generatePartId(lpuIdChain" << paramSeparator;
		fnBody << "partId)"  << stmtSeparator;
		fnBody << quadIndent << indent << "rootContainer->insertPart(*dimOrder" << paramSeparator;
		fnBody << "segmentTag" << paramSeparator << "partId)" << stmtSeparator;
		fnBody << quadIndent << "}\n";
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
	fnBody << indent << "Assert(distributionMap != NULL)" << stmtSeparator;
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
	fnBody << varName << "Space" << receiverAllocatorLpsName << "Config"<< "\")->";
	fnBody << "generateStateFulVersion()" << stmtSeparator;
	fnBody << indent << "int confinementLps = Space_" << commCharacter->getConfinementSpace()->getName();
	fnBody << stmtSeparator << indent;
	fnBody << "PartIdContainer *senderPartTree = NULL" << stmtSeparator;
	fnBody << indent <<  "DataItems *senderDataItems = taskData->getDataItemsOfLps(\"";
	fnBody << senderAllocatorLpsName << "\"" << paramSeparator;
	fnBody << "\"" << varName << "\")" << stmtSeparator;
	fnBody << indent << "if(senderDataItems != NULL) senderPartTree = ";
	fnBody << "senderDataItems->getPartIdContainer()" << stmtSeparator;
	fnBody << indent << "PartIdContainer *receiverPartTree = NULL" << stmtSeparator;
	fnBody << indent << "DataItems *receiverDataItems = taskData->getDataItemsOfLps(\"";
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

	fnBody << indent << "Assert(confinementConfig != NULL)" << stmtSeparator;
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
	List<CommunicationCharacteristics*> *commWithDataMovements = new List<CommunicationCharacteristics*>;
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		CommunicationCharacteristics *currComm = commCharacterList->Nth(i);
		if (!currComm->isCommunicationRequired()) continue;

		commWithDataMovements->Append(currComm);
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

	// return the communication characteristics involving data movements to be used by later code generation functions
	return commWithDataMovements;
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
	fnBody << indent << "Assert(dataExchangeList != NULL)" << stmtSeparator;
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

void generateScalarCommmunicatorFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
                Space *rootLps, CommunicationCharacteristics *commCharacter) {
	
	const char *dependencyName = commCharacter->getSyncRequirement()->getDependencyArc()->getArcName();
	
	decorator::writeSubsectionHeader(headerFile, dependencyName);
	decorator::writeSubsectionHeader(programFile, dependencyName);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;
	
	fnHeader << "getCommunicatorFor_" << dependencyName << "(SegmentState *localSegment" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "List<SegmentState*> *segmentList" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "TaskData *taskData" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "CommStatistics *commStat" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "TaskGlobals *taskGlobals)";

	fnBody << "{\n\n";
	
	// enlist an entry for this communicator in the communication-statistics collector
	fnBody << indent << "commStat->enlistDependency(\"" << dependencyName << "\")" << stmtSeparator;

	const char *varName = commCharacter->getVarName();
	DataStructure *structure = rootLps->getStructure(varName);
	Type *varType = structure->getType();
	const char *senderSyncLpsName = commCharacter->getSenderSyncSpace()->getName();
	const char *receiverSyncLpsName = commCharacter->getReceiverSyncSpace()->getName();
	
	// first check if the current synchronization is relevent to the segment executing this code; if not then return NULL
	fnBody << indent << "if (!taskData->hasDataForLps(\"" << senderSyncLpsName << "\") ";
	fnBody << "&& !taskData->hasDataForLps(\"" << receiverSyncLpsName << "\")) return NULL" << stmtSeparator;

	// determine the size of the scalar variable and the tag of the local segment
	fnBody << indent << "int dataSize = sizeof(" << varType->getCType() << ")" << stmtSeparator;
	fnBody << indent << "int localSegmentTag = localSegment->getPhysicalId()" << stmtSeparator;

	// determine the number of PPUs (threads) within the current segment that will participate in the communication
	fnBody << indent << "int localSenderPpus = localSegment->getPpuCountForLps(Space_";
	fnBody << senderSyncLpsName << ")" << stmtSeparator;
	fnBody << indent << "int localReceiverPpus = localSegment->getPpuCountForLps(Space_";
	fnBody << receiverSyncLpsName << ")" << stmtSeparator;

	// if there is no local sender or receiver PPU then again this communicator is not meaningful for the current segment
	fnBody << indent << "if(!localSegment->computeStagesInLps(Space_" << senderSyncLpsName << ") ";
	fnBody << '\n' << tripleIndent << "&& !localSegment->computeStagesInLps(Space_";
	fnBody << receiverSyncLpsName << ")) return NULL" << stmtSeparator;

	// declare a communicator and instantiate it with appropriate communicator type
	fnBody << '\n' << indent << "ScalarCommunicator *communicator = NULL" << stmtSeparator;

	// instantiate a communicator of appropriate type 
	SyncRequirement *syncRequirement = commCharacter->getSyncRequirement();
	if (dynamic_cast<ReplicationSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new ScalarReplicaSyncCommunicator(localSegmentTag";
	} else if (dynamic_cast<UpPropagationSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new ScalarUpSyncCommunicator(localSegmentTag";
	} else {
		fnBody << indent << "communicator = new ScalarDownSyncCommunicator(localSegmentTag";
	}
	fnBody << paramSeparator << '\n' << indent << doubleIndent;
	fnBody << '"' << dependencyName <<  '"' << paramSeparator;
	fnBody << "localSenderPpus" << paramSeparator << "localReceiverPpus" << paramSeparator;
	fnBody << "dataSize)" << stmtSeparator;

	// check if the allocation was successful and set up the data buffer reference in the communicator for the scalar
	fnBody << indent << "Assert(communicator != NULL)" << stmtSeparator;
	fnBody << indent << "communicator->setDataBufferReference(&(taskGlobals->" << varName << "))" << stmtSeparator;
	fnBody << indent << "communicator->setCommStat(commStat)" << stmtSeparator;

	fnBody << indent << "return communicator" << stmtSeparator;
	fnBody << "}\n";
	
	headerFile << "Communicator *" << fnHeader.str() << stmtSeparator;
	programFile << "\nCommunicator *" << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();
}

void generateArrayCommmunicatorFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
                Space *rootLps, CommunicationCharacteristics *commCharacter) {
	
	const char *dependencyName = commCharacter->getSyncRequirement()->getDependencyArc()->getArcName();
	const char *varName = commCharacter->getVarName();
	const char *senderAllocatorLpsName = commCharacter->getSenderDataAllocatorSpace()->getName();
	const char *receiverAllocatorLpsName = commCharacter->getReceiverDataAllocatorSpace()->getName();
	
	const char *signalerLpsName = commCharacter->getSenderSyncSpace()->getName();
	const char *waitingLpsName = commCharacter->getReceiverSyncSpace()->getName();
	if (commCharacter->doesWaitSignalDifferFromSyncLpses()) {
		waitingLpsName = commCharacter->getWaitingSpace()->getName();
		signalerLpsName = commCharacter->getSignalerSpace()->getName();
	}
	
	DataStructure *structure = rootLps->getStructure(varName);
	ArrayType *varType = (ArrayType*) structure->getType();
	const char *elementTypeName = varType->getTerminalElementType()->getCType();
	
	decorator::writeSubsectionHeader(headerFile, dependencyName);
	decorator::writeSubsectionHeader(programFile, dependencyName);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;

	fnHeader << "getCommunicatorFor_" << dependencyName << "(SegmentState *localSegment" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "TaskData *taskData" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "CommStatistics *commStat" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "PartDistributionMap *distributionMap)";

	fnBody << "{\n\n";

	// enlist an entry for this communicator in the communication-statistics collector
	fnBody << indent << "commStat->enlistDependency(\"" << dependencyName << "\")" << stmtSeparator;

	// create confinement configuration object for the dependency first
	fnBody << indent << "int localSegmentTag = localSegment->getPhysicalId()" << stmtSeparator;
	fnBody << indent << "ConfinementConstructionConfig *ccConfig = getConfineConstrConfigFor_";
	fnBody << dependencyName << "(" << '\n' << tripleIndent <<  "taskData" << paramSeparator;
	fnBody << '\n' << tripleIndent << "partConfigMap" << paramSeparator;
	fnBody << '\n' << tripleIndent << "localSegmentTag" << paramSeparator;
	fnBody << '\n' << tripleIndent << "distributionMap)" << stmtSeparator;
	fnBody << indent << "ccConfig->configurePaddingInPartitionConfigsForReadWrite()" << stmtSeparator;
	
	// then retrieve all data exchanges applicable for the current segments for this dependency
	fnBody << indent << "struct timeval start" << stmtSeparator;
        fnBody << indent << "gettimeofday(&start, NULL)" << stmtSeparator;
	fnBody << indent << "List<DataExchange*> *dataExchangeList = getDataExchangeListFor_";
	fnBody << dependencyName << "(taskData" << paramSeparator;
	fnBody << '\n' << tripleIndent << "partConfigMap" << paramSeparator;
	fnBody << '\n' << tripleIndent << "localSegmentTag" << paramSeparator;
	fnBody << '\n' << tripleIndent << "distributionMap)" << stmtSeparator;

	// if there is no data-exchanges in the list then the current segment will not participate in any communication involving 
	// this data dependency
	fnBody << indent << "if (dataExchangeList == NULL || dataExchangeList->NumElements() == 0) return NULL";
	fnBody << stmtSeparator << '\n';

	// otherwise log the time spent on creating the data exchange list
	fnBody << indent << "struct timeval middle" << stmtSeparator;
        fnBody << indent << "gettimeofday(&middle, NULL)" << stmtSeparator;
	fnBody << indent << "commStat->addConfinementConstrTime(\"" << dependencyName << "\"" << paramSeparator;
	fnBody << "start" << paramSeparator << "middle)" << stmtSeparator;

	// create a synchronization configuration object so that would be created buffers can coordinate with operating memory
	fnBody << indent << "DataPartsList *senderDataParts = NULL" << stmtSeparator;
	fnBody << indent <<  "DataItems *senderDataItems = taskData->getDataItemsOfLps(\"";
	fnBody << senderAllocatorLpsName << "\"" << paramSeparator;
	fnBody << "\"" << varName << "\")" << stmtSeparator;
	fnBody << indent << "if(senderDataItems != NULL) senderDataParts = ";
	fnBody << "senderDataItems->getPartsList()" << stmtSeparator;
	fnBody << indent << "DataPartsList *receiverDataParts = NULL" << stmtSeparator;
	fnBody << indent << "DataItems *receiverDataItems = taskData->getDataItemsOfLps(\"";
	fnBody << receiverAllocatorLpsName << "\"" << paramSeparator;
	fnBody << "\"" << varName << "\")" << stmtSeparator;
	fnBody << indent << "if(receiverDataItems != NULL) receiverDataParts = ";
	fnBody << "receiverDataItems->getPartsList()" << stmtSeparator;
	fnBody << indent << "SyncConfig *syncConfig = new SyncConfig(ccConfig" << paramSeparator;
	fnBody << '\n' << indent << doubleIndent;
	fnBody << "senderDataParts" << paramSeparator << "receiverDataParts" << paramSeparator;
	fnBody << "sizeof(" << elementTypeName << "))" << stmtSeparator;
	fnBody << indent << "Assert(syncConfig != NULL)" << stmtSeparator << '\n';

	// find out the number of sender and receiver PPUs within current segment
	fnBody << indent << "int localSenderPpus = localSegment->getPpuCountForLps(Space_";
	fnBody << signalerLpsName << ")" << stmtSeparator;
	fnBody << indent << "int localReceiverPpus = localSegment->getPpuCountForLps(Space_";
	fnBody << waitingLpsName << ")" << stmtSeparator << '\n';

	// determine if the data structure has multiple versions; this will tell if we can preprocessed buffers that keep track
	// of operating memory addresses to populate (and vice versa) elements of the communication buffer for a data exchange
	// note that the default version count is 0 
	int versionCount = structure->getVersionCount();
	const char *bufferTypePrefix = (versionCount > 0) ? "SwiftIndexMapped" : "Preprocessed";

	// instantiate a list of communication buffers to add virtual/physical communication buffers into it for data exchanges
	// based on whether or not the exchange demands cross-segments communication 
	fnBody << indent << "List<CommBuffer*> *bufferList = new List<CommBuffer*>" << stmtSeparator;
	fnBody << indent << "Assert(bufferList != NULL)" << stmtSeparator;
	// also instanciate a vector for tracking all segments that do communication for the current data dependency within the
	// confinements that are relevant to the current segment
	fnBody << indent << "std::vector<int> *participantTags = new std::vector<int>" << stmtSeparator;
	fnBody << indent << "Assert(participantTags != NULL)" << stmtSeparator;
	fnBody << indent << "for (int i = 0; i < dataExchangeList->NumElements(); i++) {\n\n";
	fnBody << doubleIndent << "DataExchange *exchange = dataExchangeList->Nth(i)" << stmtSeparator;
	// put the participant segments in the participants list and skip this data exchange if the local segment is not 
	// participating in the data exchange
	fnBody << doubleIndent << "std::vector<int> senderTags";
	fnBody << " = exchange->getSender()->getSegmentTags()" << stmtSeparator;
	fnBody << doubleIndent << "binsearch::addThoseNotExist(participantTags" << paramSeparator;
	fnBody << "&senderTags)" << stmtSeparator;
	fnBody << doubleIndent << "std::vector<int> receiverTags";
	fnBody << " = exchange->getReceiver()->getSegmentTags()" << stmtSeparator;
	fnBody << doubleIndent << "binsearch::addThoseNotExist(participantTags" << paramSeparator;
	fnBody << "&receiverTags)" << stmtSeparator;
	fnBody << doubleIndent << "if (!exchange->involvesLocalSegment(localSegmentTag)) continue"; 
	fnBody << stmtSeparator << "\n";
	// otherwise continue and create an appropriate communication buffer for the exchange and add that in the list of buffers
	fnBody << doubleIndent << "CommBuffer *buffer = NULL" << stmtSeparator;
	fnBody << doubleIndent << "if (exchange->isIntraSegmentExchange(localSegmentTag)) {\n";
	fnBody << tripleIndent << "buffer = new " << bufferTypePrefix << "VirtualCommBuffer(";
	fnBody << "exchange" << paramSeparator;
	fnBody << "syncConfig" << ")" << stmtSeparator;
	fnBody << doubleIndent << "} else {\n";
	fnBody << tripleIndent << "buffer = new " << bufferTypePrefix << "PhysicalCommBuffer(";
	fnBody << "exchange" << paramSeparator;
	fnBody << "syncConfig" << ")" << stmtSeparator;
	fnBody << doubleIndent << "}\n";
	fnBody << doubleIndent << "Assert(buffer != NULL)" << stmtSeparator;
	fnBody << doubleIndent << "bufferList->Append(buffer)" << stmtSeparator;
	fnBody << indent << "}\n";

	// if there is no buffer in the buffer list then this segment does not participate in dependency resolution at all; thus
	// returns a NULL communicator in that case
	fnBody << indent << "if (bufferList->NumElements() == 0) return NULL" << stmtSeparator;

	// otherwise log the time spent on creating the communication buffers
	fnBody << indent << "struct timeval end" << stmtSeparator;
        fnBody << indent << "gettimeofday(&end, NULL)" << stmtSeparator;
	fnBody << indent << "commStat->addBufferSetupTime(\"" << dependencyName << "\"" << paramSeparator;
	fnBody << "middle" << paramSeparator << "end)" << stmtSeparator;

	// check the type of synchronization and create a communicator appropriate for that type
	fnBody << '\n' << indent << "Communicator *communicator = NULL" << stmtSeparator;
	SyncRequirement *syncRequirement = commCharacter->getSyncRequirement();
	if (dynamic_cast<ReplicationSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new ReplicationSyncCommunicator(localSegmentTag";
	} else if (dynamic_cast<GhostRegionSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new GhostRegionSyncCommunicator(localSegmentTag";
	} else if (dynamic_cast<CrossPropagationSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new CrossSyncCommunicator(localSegmentTag";
	} else if (dynamic_cast<UpPropagationSync*>(syncRequirement) != NULL) {
		fnBody << indent << "communicator = new UpSyncCommunicator(localSegmentTag";
	} else {
		fnBody << indent << "communicator = new DownSyncCommunicator(localSegmentTag";
	}
	fnBody << paramSeparator << '\n' << indent << doubleIndent;
	fnBody << '"' << dependencyName << '"' << paramSeparator;
	fnBody << "localSenderPpus" << paramSeparator << "localReceiverPpus" << paramSeparator;
	fnBody << "bufferList)" << stmtSeparator;
	fnBody << indent << "Assert(communicator != NULL)" << stmtSeparator;
	fnBody << indent << "communicator->setParticipants(participantTags)" << stmtSeparator;
	fnBody << indent << "communicator->setCommStat(commStat)" << stmtSeparator;

	fnBody << indent << "return communicator" << stmtSeparator;
	fnBody << "}\n";
	
	headerFile << "Communicator *" << fnHeader.str() << stmtSeparator;
	programFile << "\nCommunicator *" << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();
}

void generateAllCommunicators(const char *headerFileName,
                const char *programFileName,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList) {
	
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for generating communicator functions";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout <<  "Unable to open output header file for generating communicator functions";
                std::exit(EXIT_FAILURE);
        }

	const char *initials = string_utils::getInitials(taskDef->getName());
        initials = string_utils::toLower(initials);
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
		
	std::cout << "\tGenerating functions for communicator for data synchronizations\n";
	
	const char *message = "functions to generate communicators for exchanging data to resolve dependencies";
	decorator::writeSectionHeader(headerFile, message);
	decorator::writeSectionHeader(programFile, message);

	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		CommunicationCharacteristics *commCharacter = commCharacterList->Nth(i);
		const char *varName = commCharacter->getVarName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array == NULL) {
			generateScalarCommmunicatorFn(headerFile, 
					programFile, initials, rootLps, commCharacter);
		} else {
			generateArrayCommmunicatorFn(headerFile, 
					programFile, initials, rootLps, commCharacter);
		}	
	}

	programFile.close();
	headerFile.close();
}

void generateCommunicatorMapFn(const char *headerFileName,
                const char *programFileName,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList) {
	
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open()) {
                std::cout << "Unable to open output program file for generating a communicator map";
                std::exit(EXIT_FAILURE);
        }
        if (!headerFile.is_open()) {
                std::cout <<  "Unable to open output header file for generating a communicator map";
                std::exit(EXIT_FAILURE);
        }

	const char *initials = string_utils::getInitials(taskDef->getName());
        initials = string_utils::toLower(initials);
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();

	decorator::writeSubsectionHeader(headerFile, "communicator map");
	decorator::writeSubsectionHeader(programFile, "communicator map");

	std::ostringstream fnHeader;
	std::ostringstream fnBody;

	// define the function signature
	fnHeader << "generateCommunicators(SegmentState *localSegment" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "List<SegmentState*> *segmentList" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "TaskData *taskData" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "TaskGlobals *taskGlobals" << paramSeparator;
	fnHeader << '\n' << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "PartDistributionMap *distributionMap" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "CommStatistics *commStat" << paramSeparator;
	fnHeader << "\n" << doubleIndent << "std::ofstream &logFile)";

	fnBody << "{\n\n";

	// instantiate a communicator map
	fnBody << indent << "Hashtable<Communicator*> *communicatorMap = new Hashtable<Communicator*>";
	fnBody << stmtSeparator;
	fnBody << indent << "Assert(communicatorMap != NULL)" << stmtSeparator;
	
	// determine the number of segments in the machine to be used for setting up communication buffer tags
	fnBody << indent << "int segmentCount = Total_Threads / Threads_Per_Segment" << stmtSeparator << "\n";

	// iterate over the list of communication characteristics and invoke appropriate function to create a communicator
	// each entry in the list
	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		
		CommunicationCharacteristics *commCharacter = commCharacterList->Nth(i);
		SyncRequirement *syncReq = commCharacter->getSyncRequirement();
		const char *dependencyName = syncReq->getDependencyArc()->getArcName();
		const char *varName = commCharacter->getVarName();
		DataStructure *structure = rootLps->getStructure(varName);
		Type *varType = structure->getType();
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		
		fnBody << indent << "Communicator *communicator" << i << " = getCommunicatorFor_";
		fnBody << dependencyName << "(";
		if (array == NULL) {
			fnBody << "localSegment" << paramSeparator;
			fnBody << '\n' << indent << doubleIndent;
			fnBody << "segmentList" << paramSeparator;
			fnBody << "taskData" << paramSeparator;
			fnBody << "commStat" << paramSeparator;
			fnBody << "taskGlobals";
		} else {
			fnBody << "localSegment" << paramSeparator;
			fnBody << '\n' << indent << doubleIndent;
			fnBody << "taskData" << paramSeparator;
			fnBody << "partConfigMap" << paramSeparator;
			fnBody << "commStat" << paramSeparator;
			fnBody << "distributionMap";
		}
		fnBody << ")" << stmtSeparator;
		fnBody << indent << "if (communicator" << i << " != NULL) {\n";
		fnBody << doubleIndent << "communicator" << i <<  "->setLogFile(&logFile)" << stmtSeparator;
		fnBody << doubleIndent << "communicator" << i << "->setupBufferTags(" << i + 1;
		fnBody << paramSeparator << "segmentCount)" << stmtSeparator;
		fnBody << doubleIndent << "communicator" << i;
		if (array == NULL) {
			// scalar communicator needs not to bother about segments that they do not directly interact with
			fnBody << "->setupCommunicator(false)";
		} else {
			// for arrays, only up-propagation and down-propagation syncs need to be mindfull of communications
			// among other segments because of the collective mode of communication
			UpPropagationSync *upSync = dynamic_cast<UpPropagationSync*>(syncReq);
			DownPropagationSync *downSync = dynamic_cast<DownPropagationSync*>(syncReq);
			if (upSync != NULL || downSync != NULL) {
				fnBody << "->setupCommunicator(true)";
			} else {
				fnBody << "->setupCommunicator(false)";
			}
		}
		fnBody << stmtSeparator;
		fnBody << doubleIndent << "communicatorMap->Enter(\"" << dependencyName << "\"" << paramSeparator;
		fnBody << "communicator" << i << ")" << stmtSeparator;

		// if the current segment does not participate in communication for the underlying dependency then exclude
		// it from any communication resource setup that needs participation from all segments (both communicating
		// and non communicating) by invoking a static function in the communicator class.
		if (commCharacter->shouldAllocateGroupResources()) {		
			fnBody << indent << "} else {\n";
			fnBody << doubleIndent << "Communicator::excludeOwnselfFromCommunication(";
			fnBody << '"' << dependencyName << '"' << paramSeparator;
			fnBody << '\n' << quadIndent;
			fnBody << "localSegment->getPhysicalId()" << paramSeparator;
			fnBody << "logFile)" << stmtSeparator;
		}
		fnBody << indent << "}\n";

		// at this time set up the index for the sync requirement to maintain a runtime counter for its usage
		syncReq->setIndex(i);	
	}

	// return the map
	fnBody << '\n' << indent << "return communicatorMap" << stmtSeparator;
	fnBody << "}";	

	headerFile << "Hashtable<Communicator*> *" << fnHeader.str() << stmtSeparator;
	programFile << "\nHashtable<Communicator*> *" << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();

	programFile.close();
	headerFile.close();
}

void generateCommunicationExcludeFn(const char *headerFileName,
                const char *programFileName,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList) {
	
	std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open output file for generating a communication exclude routine";
                std::exit(EXIT_FAILURE);
        }

	const char *initials = string_utils::getInitials(taskDef->getName());
        initials = string_utils::toLower(initials);
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();

	const char *message = "communication exclude routine";
	decorator::writeSubsectionHeader(headerFile, message);
	decorator::writeSubsectionHeader(programFile, message);

	std::ostringstream fnHeader;
	std::ostringstream fnBody;

	fnHeader << "excludeFromAllCommunication(";
	fnHeader << "int segmentId" << paramSeparator;
	fnHeader << "std::ofstream &logFile";
	fnHeader << ")";

	fnBody << "{\n\n";
	
	for (int i = 0; i < commCharacterList->NumElements(); i++) {
		CommunicationCharacteristics *commCharacter = commCharacterList->Nth(i);
		SyncRequirement *syncReq = commCharacter->getSyncRequirement();
		const char *dependencyName = syncReq->getDependencyArc()->getArcName();
		if (commCharacter->shouldAllocateGroupResources()) {		
			fnBody << indent << "Communicator::excludeOwnselfFromCommunication(";
			fnBody << '"' << dependencyName << '"' << paramSeparator;
			fnBody << "segmentId" << paramSeparator;
			fnBody << "logFile)" << stmtSeparator;
		}
	}

	fnBody << "}\n";
	
	headerFile << "void " << fnHeader.str() << stmtSeparator;
	programFile << "\nvoid " << initials << "::";
	programFile << fnHeader.str() << " " << fnBody.str();

	programFile.close();
	headerFile.close();
}
