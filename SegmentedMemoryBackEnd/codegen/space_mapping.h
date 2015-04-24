#ifndef _H_space_mapping
#define	_H_space_mapping

#include "../utils/list.h"
#include "../semantics/task_space.h"
#include <iostream>

/* object definition to keep track of the configuration of a PCubeS space */
class PPS_Definition {
  public:
	int id;
	const char *name;
	int units;
	/* We need a variable that designate the PPS representing CPU cores. This is required 
	   to manage thread affinity. For the sake of identification, the current requirement 
           is that in the PCubeS description file, the space correspond to CPU cores should be 
	   marked as '*' besides its name.
	*/
	bool coreSpace;
};

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

/* function definition to generate constants corresponds to LPSes */
void generateLPSConstants(const char *outputFile, MappingNode *mappingRoot);

/* function definition to generate the thread counts for all PPSes */
void generatePPSCountConstants(const char *outputFile, List<PPS_Definition*> *pcubesConfig); 

/* 
   We need to know what processor Id of the target hardware correspond to what actual physical
   unit in the hardware. Otherwise, there may be a mismatch in the expected behavior of the
   code from its actual performance as processor numbering does not necessarily happen in an
   increasing order. This functions parse the processor description file, created by inpecting
   the /proc/cpuinfo file, and generate an array that sort processor Ids so that we can get
   the physical unit intended for a virtual processor id.
*/
void generateProcessorOrderArray(const char *outputFile, const char *processorFile);

/* function definition to generate get-partition-count() routine for any given space */
List<PartitionParameterConfig*> *generateLPUCountFunction(std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials,
		Space *space, 
		List<Identifier*> *partitionArgs);

/* function that calls the above function repeatedly to generate get-partition-count() 
   functions for all un-partitioned spaces.
*/
Hashtable<List<PartitionParameterConfig*>*> *generateLPUCountFunctions(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot, 
		List<Identifier*> *partitionArgs);

/* function definition to generate routine for retrieving dimensions metadata for an array
   in a given space given the id of the LPU for which the routine is invoked 
*/
List<int> *generateGetArrayPartForLPURoutine(Space *space, 
		ArrayDataStructure *array,
		std::ostream &headerFile,  
		std::ofstream &programFile, 
		const char *initials, 
		List<Identifier*> *partitionArgs);

/* function that calls the above function for all arrays partitioned in different spaces */
Hashtable<List<int>*> *generateAllGetPartForLPURoutines(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot, 
		List<Identifier*> *partitionArgs);

#endif
