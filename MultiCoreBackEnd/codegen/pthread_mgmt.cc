#include "pthread_mgmt.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

void generateArgStructForPthreadRunFn(const char *taskName, const char *fileName) {
	
	std::cout << "Generating a data structure for PThreads execution \n";

        std::string stmtSeparator = ";\n";
        std::string stmtIndent = "\t";
        std::ofstream programFile;

        programFile.open (fileName, std::ofstream::out | std::ofstream::app);
        if (programFile.is_open()) {
                programFile << "/*-----------------------------------------------------------------------------------\n";
                programFile << "Data structure and function for Pthreads\n";
                programFile << "------------------------------------------------------------------------------------*/\n\n";
        } else {
                std::cout << "Unable to open output program file";
                std::exit(EXIT_FAILURE);
        }

	programFile << "class PThreadArg {\n";
	programFile << "  public:\n";
	programFile << stmtIndent << "const char *taskName" << stmtSeparator;
	programFile << stmtIndent << "ArrayMetadata *metadata" << stmtSeparator;
	programFile << stmtIndent << "TaskGlobals *taskGlobals" << stmtSeparator;
	programFile << stmtIndent << "ThreadLocals *threadLocals" << stmtSeparator;
	programFile << stmtIndent << string_utils::getInitials(taskName);
	programFile << "Partition partition" << stmtSeparator;
	programFile << stmtIndent << "ThreadStateImpl *threadState" << stmtSeparator;
	programFile << "};\n";

	programFile.close();
}

void generatePThreadRunFn(const char *headerFileName, const char *programFileName, const char *initials) {
	
	std::cout << "Generating run function for PThreads execution \n";

	std::string stmtSeparator = ";\n";
        std::string stmtIndent = "\t";
        std::ofstream programFile, headerFile;

        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

	headerFile << std::endl << "void *runPThreads(void *argument)" << stmtSeparator;
	headerFile.close();

        programFile << "/*-----------------------------------------------------------------------------------\n";
        programFile << "PThreads run function\n";
       	programFile << "------------------------------------------------------------------------------------*/\n\n";
	
	programFile << "void *" << initials << "::runPThreads(void *argument) {\n";
	
	programFile << stmtIndent << "PThreadArg *pthreadArg = (PThreadArg *) argument" << stmtSeparator;
	programFile << stmtIndent << "ThreadStateImpl *threadState = pthreadArg->threadState" << stmtSeparator;
	programFile << stmtIndent << "std::cout << \"Thread \" << threadState->getThreadNo() << \" has started\"";
	programFile << stmtSeparator;
	programFile << stmtIndent << "std::cout << \" executing task: \" << pthreadArg->taskName << std::endl";
	programFile << stmtSeparator;
	programFile << stmtIndent << "run(pthreadArg->metadata, \n";
	programFile << stmtIndent << stmtIndent << stmtIndent << "pthreadArg->taskGlobals, \n";		
	programFile << stmtIndent << stmtIndent << stmtIndent << "pthreadArg->threadLocals, \n";		
	programFile << stmtIndent << stmtIndent << stmtIndent << "pthreadArg->partition, \n";		
	programFile << stmtIndent << stmtIndent << stmtIndent << "threadState)";
	programFile << stmtSeparator;
	programFile << stmtIndent;
	programFile << "std::cout << \"Thread \" << threadState->getThreadNo() << \" has ended\" << std::endl";
	programFile << stmtSeparator;
	
	programFile << stmtIndent << "pthread_exit(NULL)" << stmtSeparator;
			
	programFile << "}\n\n";
	programFile.close();	
}
