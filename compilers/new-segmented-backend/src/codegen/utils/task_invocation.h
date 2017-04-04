/* This header file contains definitions of the routines that are used to invoke tasks from the
   coordinator program.
*/

#ifndef _H_task_invocation
#define _H_task_invocation

#include "task_generator.h"

#include "../../../../frontend/src/syntax/ast_def.h"

#include "../../../../common-libs/utils/list.h"

#include <sstream>
#include <fstream>
#include <iostream>

// this function generates and add the default include directives in the coordinator program's header
// and program files
void initiateProgramHeaders(const char *headerFile, 
		const char *programFile, 
		ProgramDef *programDef);

// this function, as its name suggests, generates a method that will prompt the user to enter values 
// for all program arguments, constructs an instant of the argument data structure, and returns it 
// to the caller
void generateRoutineToInitProgramArgs(TupleDef *programArg, 
		const char *headerFile, 
		const char *programFile);

// this is an alternative to the previous function that reads all program arguments as key, value 
// pairs from the command line
void generateRoutineToReadProgramArgs(TupleDef *programArg, 
		const char *headerFile, 
		const char *programFile);

// generate a function that serves as a task::main and handle initiation, execution, and after
// processing of tasks.
void generateTaskExecutor(TaskGenerator *taskGenerator);

// generate a main function based on the configuration of the coordinator program in IT source code
void generateMain(ProgramDef *programDef, const char *programFile);	

// function that invokes other functions listed here to generate data structures and method for
// needed to make multi-task programs work
void processCoordinatorProgram(ProgramDef *programDef, 
		const char *headerFile, 
		const char *programFile);

#endif
