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
class PCubeSModel;
class PPS_Definition;
class EnvironmentLink;
class PartitionParameterConfig;
class ReductionMetadata;

/* function definition for generating the routine that encodes the LPS hierarchy as parent pointer
   indexes in an array. This array is needed to quickly identify the ancestor LPUs when computing
   the number of LPUs and the metadata of a single LPU.
*/
void generateParentIndexMapRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for genering the routine for constructing the root LPU */
void generateRootLpuComputeRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for genering the routine fo updating/setting root LPU property in the thread
   state where the LPU has been already constructed */
void generateSetRootLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition for generating the routine for initializing all LPUs for different LPSes */
void generateInitializeLpusRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition to generate task specific implementation of compute-LPU-Count routine that 
   is part of the Thread-State object.
*/
void generateComputeLpuCountRoutine(std::ofstream &programFile, MappingNode *mappingRoot,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

/* function definition to generate task specific implementation of compute-next-LPU routine */
void generateComputeNextLpuRoutine(std::ofstream &programFile, MappingNode *mappingRoot);

/* function definition to generate a task specific implementation of the initializer of the map 
   of reduction result variables that keep track of a PPU's partial result for individual reductions.  
*/
void generateReductionResultMapCreateFn(std::ofstream &programFile, 
               	MappingNode *mappingRoot, 
               	List<ReductionMetadata*> *reductionInfos);

/* function definition for generating the task specific implementation of Thread-State class */
void generateThreadStateImpl(const char *headerFileName, const char *programFileName, 
		MappingNode *mappingRoot,
		List<ReductionMetadata*> *reductionInfos,
                Hashtable<List<PartitionParameterConfig*>*> *countFunctionsArgsConfig);

/* When a task is mapped to the hybrid PCubeS model of the hardware, LPUs are generated in batches  
   by a single thread for the PPUs of the entire machine, as opposed to each PPU controller thread 
   generating its own LPU one-by-one independently. The following function generates a routine that
   tells the batch PPU controller how many LPUs should be in the batch vectors for different LPSes. */
void generateLpuBatchVectorSizeSetupRoutine(const char *headerFile, 
		const char *programFile, 
		const char *initials, MappingNode *mappingRoot);

#endif
