/* This header file contains all functions that generate data structures and routines for communication handling.
 */

#ifndef _H_code_for_comm
#define _H_code_for_comm

#include "space_mapping.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"
#include "../static-analysis/sync_stat.h"
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

// This function generates a library function to generate a communication confinement construction configuration for
// a particular dependency arc. The configuration is later needed to determine what data will be exchanged between
// segments and between memory locations within a segment
void generateConfinementConstrConfigFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, CommunicationCharacteristics *commCharacter);

// This calls the above function to generate confinement construction configurations for different data dependencies
List<CommunicationCharacteristics*> *generateFnsForConfinementConstrConfigs(const char *headerFile, 
		const char *programFile, 
		TaskDef *taskDef, List<PPS_Definition*> *pcubesConfig);

// This function generates a library function that will return all intra and cross-segment data transfer requirements
// as part of a synchronization for a specific data-dependency
void generateFnForDataExchanges(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials, 
		Space *rootLps, CommunicationCharacteristics *commCharacter);

// This calls the above functions to generate data-exchanges lists for different dependency arcs
void generateAllDataExchangeFns(const char *headerFile,
                const char *programFile,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList);

// This function generates a functions to instantiating a communicator for synchronizing a scalar variable dependency
void generateScalarCommmunicatorFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
                Space *rootLps, CommunicationCharacteristics *commCharacter, 
		bool batchExecutionMode);	

// This function generates a functions to instantiating a communicator for synchronizing an array update dependency
void generateArrayCommmunicatorFn(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
                Space *rootLps, CommunicationCharacteristics *commCharacter, 
		bool batchExecutionMode);	

// This calls the two functions above to generate communicators for all data dependencies within a task that involve
// communications 
void generateAllCommunicators(const char *headerFile,
                const char *programFile,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList, 
		bool batchExecutionMode);

// This function generates a function for constructing a map of communicators for a segment that will be shared by
// its PPU controllers. Later when a situation for dependency resolution occurs the PPU controllers retrieve the
// communicator created for that dependency and invoke send or receive on it depending on the demand of the situation.    
void generateCommunicatorMapFn(const char *headerFile,
                const char *programFile,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList);

// If some segment is not going to participate in any computation related to a task then it should exclude itself 
// from all communicator setup operations too. This function generate a routine that the non-participating segment
// can call before exiting to inform other that it is quitting.
void generateCommunicationExcludeFn(const char *headerFile,
                const char *programFile,
                TaskDef *taskDef,
                List<CommunicationCharacteristics*> *commCharacterList);	

#endif
