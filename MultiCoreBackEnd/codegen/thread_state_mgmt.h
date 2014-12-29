#ifndef _H_thread_state_mgmt
#define _H_thread_state_mgmt

#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <iostream>
#include <fstream>

/* This header file contains all ingredients needed to extend the abstract Thread-State class that
   is needed for managing LPUs of thread for a particular task. 
*/

class MappingNode;
class PPS_Definition;
class EnvironmentLink;
class PartitionParameterConfig;

/* function definition for generating the routine that encodes the LPS hierarchy as parent pointer
   indexes in an array. This array is needed to quickly identify the ancestor LPUs when computing
   the number of LPUs and the metadata of a single LPU.
*/
void generateParentIndexMapRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition to generate task specific implementation of compute-LPU-Count routine that 
   is part of the Thread-State object.
*/
void generateComputeLpuCountRoutine(std::ofstream &programFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

/* function definition for generating the task specific implementation of Thread-State class */
void generateThreadStateImpl(const char *outputFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig,
                Hashtable<List<int>*> *lpuPartFunctionsArgsConfig);

#endif
