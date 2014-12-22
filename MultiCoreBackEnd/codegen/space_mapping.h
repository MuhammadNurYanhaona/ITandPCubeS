#ifndef _H_space_mapping
#define	_H_space_mapping

#include "../utils/list.h"
#include "../semantics/task_space.h"
#include <iostream>

/* structure definition to keep track of the configuration of a PCubeS space */
typedef struct {
	int id;
	const char *name;
	int units;
	/* We need a variable that designate the PPS representing CPU cores. This is required 
	   to manage thread affinity. For the sake of identification, the current requirement 
           is that in the PCubeS description file, the space correspond to CPU cores should be 
	   marked as '*' besides its name.
	*/
	bool coreSpace;
} PPS_Definition;

/* structure definition to identify an LPS-PPS mapping */
typedef struct {
	Space *LPS;
	PPS_Definition *PPS;
} MapEntry;

/* object definition to generate mapping hierarchy from partition and mapping configurations */
class MappingNode {
  public:
	MappingNode *parent;
	MapEntry *mappingConfig;
	int index;
	List<MappingNode*> *children;
};

/* As we generate getPartitionCount() functions from source code specification, we need to know
   which array's what dimension can be used for determining the LPU count along any dimension 
   of the concerned space. At the same time we need to know the indexes of the arguments passed
   in the partition section of the task that been used within the partition function for the
   chosen array. Instances of this  object will hold all these information during getPartition-
   Count() function generation so that appropriate parameters are been passed from the rest of
   the during count investigation.
*/
class PartitionParameterConfig {
  public:
	const char *arrayName;
	int dimensionNo;
	List<int> *partitionArgsIndexes; 	
};

/* function definition to read the PCubeS description of the hardware from a file */
List<PPS_Definition*> *parsePCubeSDescription(const char *filePath);

/* function defintion to parse the mapping configuration file */
MappingNode *parseMappingConfiguration(const char *taskName, 
		const char *filePath, 
		PartitionHierarchy *lpsHierarchy, 
		List<PPS_Definition*> *pcubesConfig);

/* function definition to generate macro definitions corresponds to LPSes */
void generateLPSMacroDefinitions(const char *outputFile, MappingNode *mappingRoot);

/* function definition to generate the thread counts for all PPSes */
void generatePPSCountMacros(const char *outputFile, List<PPS_Definition*> *pcubesConfig); 

/* function definition to generate get-partition-count() routine for any given space */
List<PartitionParameterConfig*> *generateLPUCountFunction(std::ofstream &programFile,
		Space *space, List<Identifier*> *partitionArgs);

/* function that calls the above function repeatedly to generate get-partition-count() 
   functions for all un-partitioned spaces.
*/
Hashtable<List<PartitionParameterConfig*>*> *generateLPUCountFunctions(const char *outputFile,
		MappingNode *mappingRoot, List<Identifier*> *partitionArgs);

/* function definition to generate routine for retrieving dimensions metadata for an array
   in a given space given the id of the LPU for which the routine is invoked 
*/
List<int> *generateGetArrayPartForLPURoutine(Space *space, ArrayDataStructure *array, 
		std::ofstream &programFile, List<Identifier*> *partitionArgs);

/* function that calls the above function for all arrays partitioned in different spaces */
Hashtable<List<int>*> *generateAllGetPartForLPURoutines(const char *outputFile, 
		MappingNode *mappingRoot, List<Identifier*> *partitionArgs);

/* function definition for generating the runtime library routine that will create ThreadIds */
void generateFnForThreadIdsAllocation(char *outputFile, MappingNode *mappingRoot);

#endif
