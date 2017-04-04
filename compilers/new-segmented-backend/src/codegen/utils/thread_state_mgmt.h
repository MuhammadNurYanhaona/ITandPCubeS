#ifndef _H_thread_state_mgmt
#define _H_thread_state_mgmt

/* This header file contains all ingredients needed to extend the abstract Thread-State class that
   is needed for managing LPUs of individual threads of a particular task. 
*/

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

class ReductionMetadata;
class MappingNode;
class PPS_Definition;
class EnvironmentLink;
class PartitionParameterConfig;

/* A function definition for generating the routine that encodes the LPS hierarchy as parent pointer
   indexes in an array. This array is needed to quickly identify the ancestor LPUs when computing
   the number of LPUs and the metadata for a single LPU.
*/
void generateParentIndexMapRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for genering the routine for constructing the root LPU */
void generateRootLpuComputeRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for genering the routine fo updating/setting root LPU property in the thread
   state if the LPU has already been constructed */
void generateSetRootLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for generating the routine for initializing all LPUs for different LPSes */
void generateInitializeLpusRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition to generate task specific implementation of compute-LPU-Count routine that 
   is part of the Thread-State object */
void generateComputeLpuCountRoutine(std::ofstream &programFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

/* function definition to generate task specific implementation of compute-next-LPU routine */
void generateComputeNextLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition to generate a task specific implementation of the initializer of the map 
   of reduction result variables that keeps track of a PPU's partial result for individual reductions	
*/
void generateReductionResultMapCreateFn(std::ofstream &programFile, 
		MappingNode *mappingRoot, 
		List<ReductionMetadata*> *reductionInfos);

/* function definition for generating the task specific implementation of Thread-State class */
void generateThreadStateImpl(const char *headerFileName, const char *programFileName, 
		MappingNode *mappingRoot,
		List<ReductionMetadata*> *reductionInfos,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

#endif
