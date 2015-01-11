#include "compute_translator.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../static-analysis/data_flow.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../syntax/ast_task.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstdlib>	

int parseComputation(FlowStage *currentStage, const char *initialsLower,
                const char *initialsUpper,
                std::ofstream &headerFile,
                std::ofstream &programFile, 
		int currentFnNo) {

	std::string stmtIndent = "\t";
	std::string paramSeparator = ", ";
	std::string stmtSeparator = ";\n";
	std::string paramIndent = "\n\t\t";

	// if this is a composite stage, recursively call this generator function to sub stages
	CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(currentStage);
	if (compositeStage != NULL) {
		int nextFnNo = currentFnNo;
		List<FlowStage*> *stageList = compositeStage->getStageList();
		for (int i = 0; i < stageList->NumElements(); i++) {
			FlowStage *stage = stageList->Nth(i);
			nextFnNo = parseComputation(stage, initialsLower, initialsUpper, 
					headerFile, programFile, nextFnNo);
		}
		return nextFnNo;
	} else {
		// if this is a sync stage then there is nothing to do here
		ExecutionStage *execStage = dynamic_cast<ExecutionStage*>(currentStage);
		if (execStage == NULL) return currentFnNo;
			
		headerFile << "void ";
		programFile << "void " << initialsLower << "::";

		// if no name is given by the user to this stage then give it a name
		if (strlen(execStage->getName()) == 0) {
			std::ostringstream nameStr;
			nameStr << initialsLower << "_function" << currentFnNo;
			execStage->setName(strdup(nameStr.str().c_str()));
		// else remove any white space from the name and turn it into lower case	
		} else {
			std::string stageName(execStage->getName());
			string_utils::shrinkWhitespaces(stageName);
			const char *newName = string_utils::replaceChar(stageName.c_str(), ' ', '_');
			execStage->setName(newName);
		}
		headerFile << execStage->getName() << "(";
		programFile << execStage->getName() << "(";

		// get the LPS where this function should execute to generate the LPU argument
		Space *space = execStage->getSpace();
		headerFile << "Space" << space->getName() << "_LPU lpu";
		programFile << "Space" << space->getName() << "_LPU lpu";
		
		// then add the default arrayMetadata, task-global and thread-local arguments
		headerFile << paramSeparator << paramIndent << "ArrayMetadata arrayMetadata";
		programFile << paramSeparator << paramIndent << "ArrayMetadata arrayMetadata";
		headerFile << paramSeparator << paramIndent << "TaskGlobals taskGlobals";
		programFile << paramSeparator << paramIndent << "TaskGlobals taskGlobals";
		headerFile << paramSeparator << paramIndent << "ThreadLocals threadLocals";
		programFile << paramSeparator << paramIndent << "ThreadLocals threadLocals";

		// then add a parameter for the partition arguments
		headerFile << paramSeparator << initialsUpper << "Partition partition";
		programFile << paramSeparator << initialsUpper << "Partition partition";

		// finish function declaration
		headerFile << ");\n\n";
		programFile << ") {\n";

		// create local variables for all array dimensions so that later on name-transformer
		// that add prefix/suffix to accessed global variables can work properly
		programFile << '\n' << stmtIndent << "//create local variables for array dimensions \n";
		List<const char*> *localArrays = space->getLocallyUsedArrayNames();
		for (int i = 0; i < localArrays->NumElements(); i++) {
			const char *arrayName = localArrays->Nth(i);
			ArrayDataStructure *array = (ArrayDataStructure*) space->getStructure(arrayName);
			int dimensions = array->getDimensionality();
			programFile << stmtIndent << "Dimension ";
			programFile  << arrayName << "PartDims[" << dimensions << "];\n";
			for (int j = 0; j < dimensions; j++) {
				programFile << stmtIndent;	
				programFile << arrayName << "PartDims[" << j << "] = *lpu.";
				programFile << arrayName << "PartDims[" << j << "]->storageDim;\n"; 
			}
		}

		// then invoke the translate code function in the stage to generate C++ equivalent of 
		// the code content
		execStage->translateCode(programFile); 
		
		// finish function body in the program file
		programFile << "}\n\n";
		return currentFnNo + 1;
	}
}

void generateFnsForComputation(TaskDef *taskDef, const char *headerFileName,
                const char *programFileName, const char *initials) {

	std::cout << "Generating functions for stages in the compute section\n";

        std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

        headerFile << "/*-----------------------------------------------------------------------------------\n";
        headerFile << "functions for compute stages \n";
        headerFile << "------------------------------------------------------------------------------------*/\n\n";
        programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "functions for compute stages \n";
        programFile << "------------------------------------------------------------------------------------*/\n\n";

	CompositeStage *computation = taskDef->getComputation();
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
	parseComputation(computation, initials, 
			upperInitials, headerFile, programFile, 0);	

	headerFile.close();
        programFile.close();
}
