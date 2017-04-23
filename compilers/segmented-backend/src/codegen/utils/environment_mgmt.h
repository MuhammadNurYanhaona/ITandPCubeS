#ifndef _H_environment_mgmt
#define _H_environment_mgmt

/* this header file contains all routines that generate data structures and functions for task environment and program 
 * environment management 
 */

#include "../../../../frontend/src/syntax/ast_task.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

/* functions that generate the task specific Environment subclass and the two functions a subclass needs to implement */
void generateTaskEnvironmentClass(TaskDef *taskDef, 
		const char *initials,
		const char *headerFile, const char *programFile);
void generateFnForItemsMapPreparation(TaskDef *taskDef, 
		const char *initials, std::ofstream &programFile);
void generateFnForTaskCompletionInstrs(TaskDef *taskDef, 
		const char *initials, std::ofstream &programFile);

/* function to populate fields of environmental link object (that is used to initialize all arrays' dimensions) from the 
   environment object used during a task invocation */
void generateFnToInitEnvLinksFromEnvironment(TaskDef *taskDef,
                const char *initials,
                const char *headerFile,
                const char *programFile);

/* After the programmer defined task initializer function has been invoked to determine the dimension-lengths/sizes of all
   environmental objects and data structures' partition configurations have been generated, LPS allocations for items in
   the task environment can be configured using dimension/size and partition information. This function generates a routine 
   that does the Lps Allocations configuration. */
void generateFnToPreconfigureLpsAllocations(TaskDef *taskDef,
                const char *initials,
                const char *headerFile,
                const char *programFile);

/* Non array environmental variables are accessed and updated through a task-globals object during a task's execution. 
 * Their values need to be copied back into proper task environment properties at the end of the execution. This function 
 * generates a routine that do the copying. */
void generateFnToCopyBackNonArrayVars(TaskDef *taskDef,
                const char *initials,
                const char *headerFile,
                const char *programFile);

#endif
