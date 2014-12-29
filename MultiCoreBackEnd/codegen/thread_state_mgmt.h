#ifndef _H_thread_state_mgmt
#define _H_thread_state_mgmt

#include "../utils/list.h"
#include "../utils/hashtable.h"

class MappingNode;
class PPS_Definition;
class EnvironmentLink;
class PartitionParameterConfig;

/* function definition to generate task specific implementation of compute-LPU-Count routine that 
   is part of the Thread-State object.
*/
void generateComputeLpuCountRoutine(const char *outputFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

/* function definition for generating the task specific implementation of Thread-State class */
void generateThreadStateImpl(const char *outputFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig,
                Hashtable<List<int>*> *lpuPartFunctionsArgsConfig);

#endif
