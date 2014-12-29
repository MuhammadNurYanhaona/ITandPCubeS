#include "thread_state_mgmt.h"
#include "space_mapping.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void generateParentIndexMapRoutine(std::ofstream &programFile, MappingNode *mappingRoot) {
	
        programFile << "// Construction of task specific LPS hierarchy index map\n";
	
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
	programFile << "}\n\n";
}

void generateComputeLpuCountRoutine(std::ofstream &programFile, MappingNode *mappingRoot, 
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig) {

        programFile << "// Implementation of task specific compute-LPU-Count function ";
	
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
			// declare a local variable for PPU count which is a default argument to all count functions
			functionBody << doubleIndent << "int ppuCount = ";
			functionBody << "threadIds->ppuIds[Space_" << lps->getName() << "].ppuCount";
			functionBody << statementSeparator;
			
			// create local variables for ancestor LPUs that will be needed to determine the dimension
			// arguments for the get-LPU-Count function 
			List<PartitionParameterConfig*> *paramConfigs 
					= countFunctionsArgsConfig->Lookup(lps->getName());
			if (paramConfigs == NULL) std::cout << "strange!!" << std::endl;
			Hashtable<const char*> *parentLpus = new Hashtable<const char*>;
			for (int i = 0; i < paramConfigs->NumElements(); i++) {
				PartitionParameterConfig *currentConfig = paramConfigs->Nth(i);	 
				const char *arrayName = currentConfig->arrayName;
				if (arrayName == NULL) continue;
				if (parentLpus->Lookup(arrayName) == NULL) {
					Space *parentLps = lps->getLocalStructure(
							arrayName)->getSource()->getSpace();
					std::ostringstream parentLpuStr;
					functionBody << doubleIndent;
					functionBody << "Space" << parentLps->getName() << "_LPU *";
					parentLpuStr << "space" << parentLps->getName() << "Lpu";
					functionBody << parentLpuStr.str() << " = ";
					functionBody << "(Space" << parentLps->getName() << "_LPU*) \n";
					functionBody << doubleIndent << doubleIndent;
					functionBody << "lpsStates[Space_" << parentLps->getName() << "]->lpu";
					functionBody << statementSeparator;
					parentLpus->Enter(arrayName, strdup(parentLpuStr.str().c_str()));
				}
			}
			
			// call the get-LPU-Count function with appropriate parameters
			functionBody << doubleIndent << "return getLPUsCountOfSpace" << lps->getName();
			functionBody << "(ppuCount";
			for (int i = 0; i < paramConfigs->NumElements(); i++) {
				PartitionParameterConfig *currentConfig = paramConfigs->Nth(i);	 
				const char *arrayName = currentConfig->arrayName;
				if (arrayName != NULL) {
					functionBody << parameterSeparator << "\n" << doubleIndent << doubleIndent;
					functionBody <<  "*" << parentLpus->Lookup(arrayName) << "->" << arrayName;
					functionBody << "PartDims[" << currentConfig->dimensionNo - 1;
					functionBody << "]->partitionDim"; 
				}
				// pass any partition arguments used by current count function
				for (int j = 0; j < currentConfig->partitionArgsIndexes->NumElements(); j++) {
					int index = currentConfig->partitionArgsIndexes->Nth(j);
					functionBody << parameterSeparator << "\n" << doubleIndent << doubleIndent;
					functionBody << "partitionArgs[" << index << "]";		
				}
			}
			functionBody << ")" << statementSeparator; 
		}
		functionBody << singleIndent << "}\n";
	}
	
	functionBody << singleIndent << "return NULL" << statementSeparator;
	functionBody << "}\n";
	
	programFile << std::endl << functionHeader.str() << " " << functionBody.str();
	programFile << std::endl;
}

void generateThreadStateImpl(const char *outputFile, MappingNode *mappingRoot, 
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig,
                Hashtable<List<int>*> *lpuPartFunctionsArgsConfig) {
	
	std::ofstream programFile;
	programFile.open (outputFile, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------" << std::endl;
                programFile << "Thread-State implementation class for the task" << std::endl;
                programFile << "------------------------------------------------------------------------------------*/" << std::endl;
	}
	else std::cout << "Unable to open output program file";
	
	// write the common class definition from the sample file in the output program
	programFile << std::endl;
	std::string line;
        std::ifstream classDefinitionFile("codegen/thread-state-class-def.txt");
	if (classDefinitionFile.is_open()) {
                while (std::getline(classDefinitionFile, line)) {
			programFile << line << std::endl;
		}
		programFile << std::endl;
		classDefinitionFile.close();
	}
	else std::cout << "Unable to open common include file";

	// construct the index array that encode the LPS hierarchy for this task
	generateParentIndexMapRoutine(programFile, mappingRoot);
	// then call the get-LPU-Count function generator method for class specific implementation
	generateComputeLpuCountRoutine(programFile, mappingRoot, countFunctionsArgsConfig);
 
	programFile.close();
}
