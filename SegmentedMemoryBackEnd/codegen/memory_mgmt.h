#ifndef _H_memory_mgmt
#define _H_memory_mgmt

/* this header file generates all runtime data structures and functions related to memory management */

#include "../semantics/task_space.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <fstream>
#include <sstream>

/* generates a function that will return the data-partition-config for an array for a particular LPS  */
void genRoutineForDataPartConfig(std::ofstream &headerFile,
		std::ofstream &programFile,
                const char *initials,
		Space *lps,
		ArrayDataStructure *array);

/* generates data-partition-config generation functions for relevant structures in all LPSes of a task */
void genRoutinesForTaskPartitionConfigs(const char *headerFile,
                const char *programFile,
                const char *initials,
		PartitionHierarchy *hierarchy);

/* generates a routine that collect data-partition-configs for different data structures within an LPS 
   to produce an LPS configuration that will be contacted during task execution to generate LPUs. To be
   effective there must be at least one data structure that will be allocated for the LPs. 
*/
void genRoutineForLpsContent(std::ofstream &headerFile,
                std::ofstream &programFile,
                const char *initials,
		Space *lps);

/* generates a routine to produce a map object that keep track of all allocations been done for a task's 
   computations. Note that it excludes memory allocation for communication, that will be dealt separately.  
*/
void genTaskMemoryConfigRoutine(const char *headerFile,
                const char *programFile,
                const char *initials, 
		PartitionHierarchy *hierarchy);

#endif
