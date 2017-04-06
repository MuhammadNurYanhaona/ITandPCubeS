#ifndef _H_fn_generator
#define _H_fn_generator

// this header file contains all library routines needed for generating code for user defined IT functions

#include "../../../../common-libs/utils/list.h"
#include "../../../../frontend/src/syntax/ast_def.h"

#include <fstream>

// this routine generate all instance types of a single type-polymorphic function needed for the execution
// of an IT program
void generateFnInstances(FunctionDef *fnDef, std::ofstream &headerFile, std::ofstream &programFile);

// this routine calls the routine above to generate definitions for all functions found in an IT code
void generateFunctions(List<Definition*> *fnDefList, const char *headerFile, const char *programFile);

// this routine is used by the above routine to generate directives for library inclusions
void generateLibraryIncludes(List<Definition*> *fnDefList, 
		std::ofstream &headerFile, std::ofstream &programFile);

#endif
