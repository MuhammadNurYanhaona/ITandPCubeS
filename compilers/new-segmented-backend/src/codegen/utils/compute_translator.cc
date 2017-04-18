#include "compute_translator.h"
#include "space_mapping.h"
#include "array_assignment.h"
#include "code_constant.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"
#include "../../../../common-libs/utils/decorator_utils.h"

#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/semantics/scope.h"
#include "../../../../frontend/src/semantics/task_space.h"
#include "../../../../frontend/src/semantics/computation_flow.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstdlib>
#include <deque>	

int parseComputation(FlowStage *currentStage, const char *initialsLower,
                const char *initialsUpper,
                std::ofstream &headerFile,
                std::ofstream &programFile, 
		int currentFnNo) {

	// set the context for code translation to task
	codecntx::enterTaskContext();

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
		StageInstanciation *execStage = dynamic_cast<StageInstanciation*>(currentStage);
		if (execStage == NULL) return currentFnNo;

		headerFile << std::endl;
		programFile << std::endl;
			
		// each function returns an integer indicating if a successful invocation has taken place
		headerFile << "int ";
		programFile << "int " << initialsLower << "::";

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
			newName = string_utils::toLower(newName);
			execStage->setName(newName);
		}
		headerFile << execStage->getName() << "(";
		programFile << execStage->getName() << "(";

		std::ostringstream paramStream;

		// get the LPS where this function should execute to generate the LPU argument
		Space *space = execStage->getSpace();
		paramStream << "Space" << space->getName() << "_LPU *lpu";
		
		// then add the default arrayMetadata, task-global and thread-local arguments
		paramStream << paramSeparator << paramIndent << "ArrayMetadata *arrayMetadata";
		paramStream << paramSeparator << paramIndent << "TaskGlobals *taskGlobals";
		paramStream << paramSeparator << paramIndent << "ThreadLocals *threadLocals";

		// then add a parameter for the map of local, partial results of reductions if the stage
		// involves some reduction
		if (execStage->hasNestedReductions()) {
			paramStream << paramSeparator << paramIndent;
			paramStream << "Hashtable<reduction::Result*> *localReductionResultMap";
		}

		// then add a parameter for the partition arguments
		paramStream << paramSeparator << paramIndent;
		paramStream << initialsUpper << "Partition partition";

		// finally, add an output file stream for logging steps of the computation stage if needed
		paramStream << paramSeparator << paramIndent << "std::ofstream &logFile";

		// finish function declaration
		headerFile << paramStream.str();
		programFile << paramStream.str();
		headerFile << ");\n\n";
		programFile << ") {\n";

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

	const char *header = "functions for compute stages";
	decorator::writeSectionHeader(headerFile, header);
	decorator::writeSectionHeader(programFile, header);

	CompositeStage *computation = taskDef->getComputation();
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
	parseComputation(computation, initials, 
			upperInitials, headerFile, programFile, 0);	

	headerFile.close();
        programFile.close();
}

void generateThreadRunFunction(TaskDef *taskDef, const char *headerFileName,
                const char *programFileName, 
		const char *initials, 
		MappingNode *mappingRoot, 
		bool involvesSynchronization,
		bool involvesReduction, int communicatorCount) {

	std::cout << "Generating the thread::run function for the task\n";
	
	const char *upperInitials = string_utils::getInitials(taskDef->getName());
        
	std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

	const char *header = "run method for thread simulating the task flow";
	decorator::writeSectionHeader(headerFile, header);
	headerFile << "\n";
	decorator::writeSectionHeader(programFile, header);
	programFile << "\n";

	// create a stream for writing the arguments default to all thread run functions
	std::ostringstream defaultArgs;
	// add the default arrayMetadata, task-global and thread-local arguments
	defaultArgs << "ArrayMetadata *arrayMetadata";
	defaultArgs << paramSeparator << paramIndent << "TaskGlobals *taskGlobals";
	defaultArgs << paramSeparator << paramIndent << "ThreadLocals *threadLocals";
	// then add a parameter for the partition arguments
	defaultArgs << paramSeparator << paramIndent << upperInitials << "Partition partition";
	// then add a parameter for the thread state variable
	defaultArgs << paramSeparator << "ThreadStateImpl *threadState";	

	// write the function signature in both header and program files
	headerFile << "void run(" << defaultArgs.str() << ");\n\n";
	programFile << "void " << initials << "::run(" << defaultArgs.str() << ") {\n";

	// first log affinity information of this thread so that later we can varify that it executed on the PPU
	// we intended
	programFile << "\n\t// log thread's affinity information\n";
	programFile << "\tthreadState->logThreadAffinity();\n";

	// set the root LPU for the thread so the computation can start
	PartitionHierarchy *hierarchy = taskDef->getPartitionHierarchy();
	Space *rootLps = hierarchy->getRootSpace();
	programFile << "\n\t// set the root LPU in the thread state so that calculation can start\n";
	programFile << "\tLPU *rootLpu = threadState->getCurrentLpu(Space_" << rootLps->getName() << ");\n";
	programFile << "\tif (rootLpu == NULL) {\n";
	programFile << "\t\tthreadState->setRootLpu(arrayMetadata);\n";
	programFile << "\t}\n";

	// if the task involves synchronization then initialize the data structure that will hold sync primitives
	// correspond to synchronizations that this thread will participate into. 
	if (involvesSynchronization) {
		programFile << "\n\t// initialize thread's sync primitives holder data structure\n";
		programFile << "\tThreadSyncPrimitive *threadSync = ";
		programFile << "getSyncPrimitives(threadState->getThreadIds());\n";
	}

	// if the task involves communications then create communicator counter variables for each data dependency
	// requiring communication
	if (communicatorCount > 0) {
		programFile << "\n\t// create counter variables for communicators\n";
		for (int i = 0; i < communicatorCount; i++) {
			programFile << "\tint commCounter" << i << " = 0;\n";
		}
	}

	// initilize the map for holding partial results of reductions
	if (involvesReduction) {
		programFile << "\n\t// initializing a map for holding local, partial results of reductions\n";
		programFile << "\tthreadState->initializeReductionResultMap();\n";
		programFile << "\tHashtable<reduction::Result*> *reductionResultsMap = ";
		programFile << "threadState->getLocalReductionResultMap();\n";
	
		programFile << "\n\t// retrieving the reduction primitives relevant to the current thread\n";
		programFile << "\tThreadIds *threadIds = threadState->getThreadIds();\n";
		programFile << "\tHashtable<void*> *rdPrimitiveMap = ";
		programFile << "getReductionPrimitiveMap(threadIds);\n";
	}

	// create a local part-dimension object for probable array dimension based range or assignment expressions
	programFile << "\n\t// create a local part-dimension object for later use\n";
	programFile << "\tPartDimension partConfig;\n";

	// create a local integer for holding intermediate values of transformed index during inclusion testing
        programFile << "\n\t// create a local transformed index variable for later use\n";
        programFile << "\tint xformIndex;\n";

	// invoke recursive flow stage invocation code to implement the logic of the run method
	CompositeStage *computation = taskDef->getComputation();
	computation->generateInvocationCode(programFile, 1, rootLps);

	// log iterator usage statistics to check the efficiency of part searching process
	programFile << "\n\t// logging iterators' efficiency\n";
	programFile << "\tthreadState->logIteratorStatistics();\n";

	// close the thread log file
	programFile << "\n\t// close thread's log file\n";
	programFile << "\tthreadState->closeLogFile();\n";
		
	// finish function body in the program file
	programFile << "}\n\n";

	headerFile.close();
        programFile.close();
}
