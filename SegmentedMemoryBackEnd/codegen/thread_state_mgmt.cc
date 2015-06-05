#include "thread_state_mgmt.h"
#include "lpu_generation.h"
#include "space_mapping.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/decorator_utils.h"
#include "../syntax/ast_task.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// This function is simple. It just copy the dimension information from the global array metadata object to the 
// partition dimensions of individual arrays. Memory for these arrays are not allocated as it is not done for 
// any LPU in any LPS. For memory allocation further analysis of the compute block is needed and the whole allocation 
// logic will be handled in a separate module.
void generateRootLpuComputeRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {
	
	std::cout << "\tGenerating routine for root LPU calculation" << std::endl;
	const char *header = "Root LPU Construction";
	decorator::writeSubsectionHeader(programFile, header);

	std::string statementSeparator = ";\n";
        std::string singleIndent = "\t";
	
	// specify the signature of the set-root-LPU function matching the virtual function in Thread-State class
	std::ostringstream functionHeader;
        functionHeader << "void ThreadStateImpl::setRootLpu(Metadata *metadata)";
        std::ostringstream functionBody;
	functionBody << "{\n\n";

	// first cast the generic metadata object into task specific array metadata object
	functionBody << singleIndent;
	functionBody << "ArrayMetadata *arrayMetadata = (ArrayMetadata*) metadata";
	functionBody << statementSeparator;
	functionBody << std::endl;

	// allocate an LPU for the root
	Space *rootLps = mappingRoot->mappingConfig->LPS;
	functionBody << singleIndent;
	functionBody << "Space" << rootLps->getName() << "_LPU *lpu = new Space";
	functionBody  << rootLps->getName() << "_LPU";
	functionBody << statementSeparator;

	// initialize each array in the root LPU
	List<const char*> *localArrays = rootLps->getLocallyUsedArrayNames();
	for (int i = 0; i < localArrays->NumElements(); i++) {
		if (i > 0) functionBody << std::endl;
		const char* arrayName = localArrays->Nth(i);
		ArrayDataStructure *array = (ArrayDataStructure*) rootLps->getLocalStructure(arrayName);
		int dimensionCount = array->getDimensionality();
		functionBody << singleIndent << "lpu->" << arrayName << " = NULL" << statementSeparator;
		std::ostringstream varName;
		varName << "lpu->" << arrayName << "PartDims";
		for (int j = 0; j < dimensionCount; j++) {
			functionBody << singleIndent << varName.str() << "[" << j << "] = ";
			functionBody << "PartDimension()" << statementSeparator;
			functionBody << singleIndent << varName.str() << "[" << j << "].partition = ";
			functionBody << "arrayMetadata->" << arrayName << "Dims[" << j << "]";
			functionBody << statementSeparator;			
			functionBody << singleIndent << varName.str() << "[" << j << "].storage = ";
			functionBody << "arrayMetadata->" << arrayName << "Dims[" << j;
			functionBody << "].getNormalizedDimension()" << statementSeparator;			
		}	
	}
	
	// store the LPU in the proper LPS state
	functionBody << std::endl;
	functionBody << singleIndent << "lpu->setValidBit(true)" << statementSeparator;	
	functionBody << singleIndent << "lpsStates[Space_" << rootLps->getName() << "]->lpu = lpu";
	functionBody << statementSeparator << "}\n";
	
	programFile << functionHeader.str() << " " << functionBody.str();
}

void generateSetRootLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {

	std::string statementSeparator = ";\n";
        std::string singleIndent = "\t";
	Space *rootLps = mappingRoot->mappingConfig->LPS;
	
	// specify the signature of the set-root-LPU function matching the virtual function in Thread-State class
	std::ostringstream functionHeader;
        functionHeader << "void ThreadStateImpl::setRootLpu(LPU *lpu)";
        std::ostringstream functionBody;
	functionBody << "{\n";

	// store the LPU in the proper LPS state
	functionBody << singleIndent << "lpu->setValidBit(true)" << statementSeparator;	
	functionBody << singleIndent << "lpsStates[Space_" << rootLps->getName() << "]->lpu = lpu";
	functionBody << statementSeparator << "}\n";
	
	programFile << std::endl;
	programFile << functionHeader.str() << " " << functionBody.str();
}

void generateInitializeLpusRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {
	
	std::cout << "\tGenerating function for initializing LPU pointers" << std::endl;

	const char *header = "State Initialization";
	decorator::writeSubsectionHeader(programFile, header);
	
	std::string statementSeparator = ";\n";
        std::string singleIndent = "\t";

	programFile << "void ThreadStateImpl::initializeLPUs() {\n";
	
	// Note that we skip the root LPU as it must be set correctly for computation to be able to make
	// progress and it is been correctly initialized by the setRootLpu() routine.
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
		programFile << singleIndent;
		programFile << "lpsStates[Space_" << lps->getName() << "]->lpu = ";
		programFile << "new Space" << lps->getName() << "_LPU";
		programFile << statementSeparator;
		programFile << singleIndent;
		programFile << "lpsStates[Space_" << lps->getName() << "]->lpu->setValidBit(false)";
		programFile << statementSeparator;
	}

	programFile << "}\n";
}

void generateParentIndexMapRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {
	
	std::cout << "\tGenerating parent pointer index map" << std::endl;
        
	const char *header = "LPS Hierarchy Trace";
	decorator::writeSubsectionHeader(programFile, header);
	
	std::string statementSeparator = ";\n";
        std::string singleIndent = "\t";
	std::ostringstream allocateStmt;
	std::ostringstream initializeStmts;

	allocateStmt << singleIndent << "lpsParentIndexMap = new int";
	allocateStmt << "[Space_Count]" << statementSeparator;
	initializeStmts << singleIndent << "lpsParentIndexMap[Space_";
	initializeStmts << mappingRoot->mappingConfig->LPS->getName();
	initializeStmts << "] = INVALID_ID" << statementSeparator;

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
		initializeStmts << singleIndent;
		initializeStmts << "lpsParentIndexMap[Space_" << lps->getName() << "] = ";
		initializeStmts << "Space_" << lps->getParent()->getName();
		initializeStmts << statementSeparator;
	}

	programFile << "void ThreadStateImpl::setLpsParentIndexMap() {\n";
	programFile << allocateStmt.str() << initializeStmts.str();
	programFile << "}\n";
}

void generateComputeLpuCountRoutine(std::ofstream &programFile, MappingNode *mappingRoot, 
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig) {

	std::cout << "\tGenerating compute LPU count function" << std::endl;
        
	const char *header = "LPU Count Function";
	decorator::writeSubsectionHeader(programFile, header);
	
	std::string statementSeparator = ";\n";
        std::string singleIndent = "\t";
	std::string doubleIndent = "\t\t";
	std::string tripleIndent = "\t\t\t";
	std::string parameterSeparator = ", ";

	// specify the signature of the compute-Next-Lpu function matching the virtual function in Thread-State class
	std::ostringstream functionHeader;
        functionHeader << "int *ThreadStateImpl::computeLpuCounts(int lpsId)";
        std::ostringstream functionBody;
	functionBody << "{\n";

	// retrieve the data partition configuration map from thread state that will be used by LPS count functions
	functionBody << singleIndent << "Hashtable<DataPartitionConfig*> *configMap = getPartConfigMap()";
	functionBody << statementSeparator;

	std::deque<MappingNode*> nodeQueue;
        nodeQueue.push_back(mappingRoot);
        while (!nodeQueue.empty()) {
                MappingNode *node = nodeQueue.front();
                nodeQueue.pop_front();
                for (int i = 0; i < node->children->NumElements(); i++) {
                        nodeQueue.push_back(node->children->Nth(i));
                }
		Space *lps = node->mappingConfig->LPS;
		// if the space is unpartitioned then we can return NULL as there is no need for a counter then
		functionBody << singleIndent << "if (lpsId == Space_" << lps->getName() << ") {\n";
		if (lps->getDimensionCount() == 0) {
			functionBody << doubleIndent << "return NULL" << statementSeparator;
		// otherwise, we have to call the appropriate get-LPU-count function generated before 
		} else {
			// create local variables for ancestor LPUs that will be needed to determine the dimension
			// arguments for the get-LPU-Count function 
			List<PartitionParameterConfig*> *paramConfigs 
					= countFunctionsArgsConfig->Lookup(lps->getName());
			if (paramConfigs == NULL) std::cout << "strange!!" << std::endl;
			Hashtable<const char*> *parentLpus = new Hashtable<const char*>;
			Hashtable<const char*> *arrayToParentLpus = new Hashtable<const char*>;
			for (int i = 0; i < paramConfigs->NumElements(); i++) {
				PartitionParameterConfig *currentConfig = paramConfigs->Nth(i);
				const char *arrayName = currentConfig->arrayName;
				if (arrayName == NULL) continue;
				Space *parentLps = lps->getLocalStructure(arrayName)->getSource()->getSpace();
				// no variable is yet created for the parent LPU so create it
				if (parentLpus->Lookup(parentLps->getName()) == NULL) {
					std::ostringstream parentLpuStr;
					functionBody << doubleIndent;
					functionBody << "Space" << parentLps->getName() << "_LPU *";
					parentLpuStr << "space" << parentLps->getName() << "Lpu";
					functionBody << parentLpuStr.str() << std::endl;
					functionBody << doubleIndent << doubleIndent;
					functionBody << " = (Space" << parentLps->getName() << "_LPU*) ";
					functionBody << "lpsStates[Space_" << parentLps->getName() << "]->lpu";
					functionBody << statementSeparator;
					arrayToParentLpus->Enter(arrayName, strdup(parentLpuStr.str().c_str()));
					parentLpus->Enter(parentLps->getName(), strdup(parentLpuStr.str().c_str()));
				// otherwise just get the reference of the already created LPU
				} else {
					arrayToParentLpus->Enter(arrayName, parentLpus->Lookup(parentLps->getName()));
				}
			}

			// call the get-LPU-Count function with appropriate parameters
			functionBody << doubleIndent << "return getLPUsCountOfSpace" << lps->getName();
			functionBody << "(configMap";
			for (int i = 0; i < paramConfigs->NumElements(); i++) {
				PartitionParameterConfig *currentConfig = paramConfigs->Nth(i);	 
				const char *arrayName = currentConfig->arrayName;
				if (arrayName != NULL) {
					functionBody << parameterSeparator << "\n" << doubleIndent << doubleIndent;
					functionBody << arrayToParentLpus->Lookup(arrayName) << "->" << arrayName;
					functionBody << "PartDims[" << currentConfig->dimensionNo - 1;
					functionBody << "].partition"; 
				}
			}
			functionBody << ")" << statementSeparator; 
		}
		functionBody << singleIndent << "}\n";
	}
	
	functionBody << singleIndent << "return NULL" << statementSeparator;
	functionBody << "}\n";
	
	programFile << std::endl << functionHeader.str() << " " << functionBody.str();
}

void generateComputeNextLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {
       
	std::cout << "\tGenerating compute next LPU function" << std::endl;

	const char *header = "LPU Construction Function";
	decorator::writeSubsectionHeader(programFile, header); 
	
	std::string stmtSeparator = ";\n";
        std::string indent = "\t";
	std::string doubleIndent = "\t\t";
	std::string tripleIndent = "\t\t\t";
	std::string paramSeparator = ", ";

	// specify the signature of the compute-Next-Lpu function matching the virtual function in Thread-State class
	std::ostringstream functionHeader;
        functionHeader << "LPU *ThreadStateImpl::computeNextLpu(int lpsId)";
        std::ostringstream functionBody;
	functionBody << "{\n";

	Space *rootLps = mappingRoot->mappingConfig->LPS; 

	// get the list of LPU ids leading to the LPU of the current LPS
	functionBody << indent << "List<int*> *lpuIdList = " << "getLpuIdChain(lpsId" << paramSeparator;
	functionBody << "Space_" << rootLps->getName() << ")" << stmtSeparator;

	// get the data partition config and task data objects to pass as default arguments to any LPU generation function
	functionBody << indent << "Hashtable<DataPartitionConfig*> *partConfigMap = getPartConfigMap()" << stmtSeparator;
	functionBody << indent << "TaskData *taskData = getTaskData()" << stmtSeparator;		

	// we skip the mapping root and start iterating with its descendents because the LPU for the root space should not 
	// change throughout the computation of the task
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
		functionBody << indent << "if (lpsId == Space_" << lps->getName() << ") {\n";

		// get the reference of the current LPU for update
		const char *lpsName = lps->getName();
		functionBody << doubleIndent;
		functionBody << "Space" << lpsName << "_LPU *currentLpu";
		functionBody << " = (Space" << lpsName << "_LPU*) ";
		functionBody << "lpsStates[Space_" << lpsName << "]->lpu";
		functionBody << stmtSeparator;

		// call appropriate LPU construction function to populate the fields of current LPU
		functionBody << doubleIndent << "generateSpace" << lpsName << "Lpu(";
		functionBody << "currentLpu" << paramSeparator;
		functionBody << "lpuIdList" << paramSeparator;
		functionBody << "partConfigMap" << paramSeparator;
		functionBody << "taskData" << ')' << stmtSeparator;
		
		// flag the current LPU as valid
		functionBody << doubleIndent << "currentLpu->setValidBit(true)" << stmtSeparator;
		functionBody << doubleIndent << "return currentLpu" << stmtSeparator;
		functionBody << indent << "}\n";
	}
	
	functionBody << indent << "return NULL" << stmtSeparator;
	functionBody << "}\n";
	
	programFile << std::endl << functionHeader.str() << " " << functionBody.str();
}

void generateThreadStateImpl(const char *headerFileName, 
		const char *programFileName, 
		MappingNode *mappingRoot, 
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig) {

	std::cout << "Generating task spacific Thread State implementation task" << std::endl;	
	std::ofstream programFile, headerFile;
	headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
	programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
		std::cout << "Unable to open program or header file";
		std::exit(EXIT_FAILURE);
	}
                
	// write the common class definition from the sample file in the header file
	const char *message = "Thread-State implementation class for the task";
	decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
	std::string line;
        std::ifstream classDefinitionFile("codegen/thread-state-class-def.txt");
	if (classDefinitionFile.is_open()) {
                while (std::getline(classDefinitionFile, line)) {
			headerFile << line << std::endl;
		}
		headerFile << std::endl;
		classDefinitionFile.close();
	} else {
		std::cout << "Unable to open common include file";
		std::exit(EXIT_FAILURE);
	}
	headerFile.close();

	decorator::writeSectionHeader(programFile, message);
	
	//---------------------------------------write the implementions of virtual functions in the program file
	// construct the index array that encode the LPS hierarchy for this task
	generateParentIndexMapRoutine(programFile, mappingRoot);
	// generate the function for creating the root LPU from array metadata information
	generateRootLpuComputeRoutine(programFile, mappingRoot);
	// generate the function for setting the root LPU constructed using some other means
	generateSetRootLpuRoutine(programFile, mappingRoot);
	// generate the function for allocating a pointer for each LPS'es LPU that will be updated over and over
	generateInitializeLpusRoutine(programFile, mappingRoot);
	// then call the compute-LPU-Count function generator method for class specific implementation
	generateComputeLpuCountRoutine(programFile, mappingRoot, countFunctionsArgsConfig);
	// then call the compute-Next-LPU function generator method for class specific implementation
	generateComputeNextLpuRoutine(programFile, mappingRoot);
	//-------------------------------------------------------------------------------------------------------
 
	programFile.close();
}
