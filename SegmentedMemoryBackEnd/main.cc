/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 */
 
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include "parser.h"
#include "scanner.h"
#include "codegen/task_invocation.h"
#include "codegen/task_generator.h"	
#include "codegen/code_generator.h"	
#include "codegen/thread_state_mgmt.h"	
#include "codegen/space_mapping.h"
#include "utils/list.h"
#include "syntax/ast.h"
#include "syntax/ast_def.h"
#include "syntax/errors.h"
#include "syntax/ast_task.h"

int main(int argc, const char *argv[]) {

	//***************************************************** Command Line Arguments Reader
	// read the input arguments and keep track of the appropriate files
	const char *sourceFile, *pcubesFile, *processorFile, *mappingFile;
	if (argc < 5) {
		std::cout << "You need to pass four files as input arguments" << std::endl;
		std::cout << "\t" << "1. The input IT program file" << std::endl;
		std::cout << "\t" << "2. The PCubeS model of the machine" << std::endl;
		std::cout << "\t" << "3. A processor description file for the machine\n";
		std::cout << "\t" << "4. The mapping configuration file" << std::endl;
		return -1;
	} else {
		sourceFile = argv[1];
		std::cout << "Source program: " << sourceFile << std::endl;
		pcubesFile = argv[2];
		std::cout << "PCubeS model: " << pcubesFile << std::endl;
		processorFile = argv[3];
		std::cout << "Processor configuration: " << processorFile << std::endl;	
		mappingFile = argv[4];
		std::cout << "Mapping Configuration: " << mappingFile << std::endl;	
	}
	// redirect standard input to the source file for the front end compiler to work
	int fileDescriptor = open(sourceFile, O_RDONLY);
	if (fileDescriptor < 0) {
		std::cout << "Could not open the source program file" << std::endl;
		return -1;
	}
	dup2(fileDescriptor, STDIN_FILENO);
	close(fileDescriptor);
	// set the default output directory and common header file parameters
	const char *outputDir = "build/";
	const char *tupleHeader = "build/tuple.h";
	// set the output header and program files for the coordinator program
	const char *coordHeader = "build/coordinator.h";
	const char *coordProgram = "build/coordinator.cc";
	//***********************************************************************************


	
	//**************************************************************** Front End Compiler
 	/* Entry point to the entire program. InitScanner() is used to set up the scanner.
	 * InitParser() is used to set up the parser. The call to yyparse() will attempt to
	 * parse a complete program from the input. 
	 */	
    	InitScanner(); InitParser(); yyparse();
	if (ReportError::NumErrors() > 0) return -1;
	// Do scope and type checking and other semantic analysis on the program definition 
	// generated by the parser
	ProgramDef::program->attachScope(NULL);
        ProgramDef::program->validateScope(NULL);
	if (ReportError::NumErrors() > 0) return -1;
	// Do static analysis analysis on the validated program
	ProgramDef::program->performStaticAnalysis();	
	if (ReportError::NumErrors() > 0) return -1;
	//***********************************************************************************



	//***************************************************************** Back End Compiler
	// parse PCubeS description of the multicore hardware
	List<PPS_Definition*> *pcubesConfig = parsePCubeSDescription(pcubesFile);
	// iterate over list of tasks and generate code for each of them in separate files
	List<TaskDef*> *taskList = ProgramDef::program->getTasks();
	for (int i = 0; i < taskList->NumElements(); i++) {
		TaskDef *taskDef = taskList->Nth(i);
		// do static analysis of the task to determine what data structure has been 
		// accessed in what LPS before code generation starts
		taskDef->getComputation()->calculateLPSUsageStatistics();
		TaskGenerator *generator = new TaskGenerator(taskDef, 
				outputDir, mappingFile, processorFile);
		generator->generate(pcubesConfig);
	}
	// generate classes for the list of tuples present in the source in a header file
	generateClassesForTuples(tupleHeader, ProgramDef::program->getTuples());
	// invoke the library handling task-invocations to generate all routines needed for 
	// multi-task management and a the main function corresponds to the coordinator 
	// program definition
	processCoordinatorProgram(ProgramDef::program, coordHeader, coordProgram);
	//***********************************************************************************
}

