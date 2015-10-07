/* This header file contains all functions that generate data structures and routines for communication handling.
 */

#ifndef _H_code_for_comm
#define _H_code_for_comm

#include "space_mapping.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

// This function generates a library function that creates and populates the distribution tree for a data structure.
// Remember that a distribution tree holds information about all independent partitioning and location of data parts
// in different segments.
void generateDistributionTreeFnForStructure(const char *varName, 
		std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials, Space *rootLps);

// This function determine what variables will need distribution trees and generate functions to create those trees
// by invoking the function above.
void generateFnsForDistributionTrees(const char *headerFile, 
		const char *programFile, 
		TaskDef *taskDef, 
		List<PPS_Definition*> *pcubesConfig);

// This function generates a library function to construct a map that will holds part distribution trees for various
// data structures. This gets invoked by the function above when appropriate
void generateFnForDistributionMap(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, List<const char*> *varList);

#endif
