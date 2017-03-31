#ifndef _H_lpu_generation
#define	_H_lpu_generation

#include "space_mapping.h"

#include "../../../common-libs/utils/list.h"
#include "../../../frontend/src/semantics/task_space.h"
#include <iostream>

/* As we generate getPartitionCount() functions from source code specification, we need to know which array's what 
   dimension can be used for determining the LPU count along any dimension of the concerned LPS. Instances of this  
   object hold array-dimension and alignment information during getPartitionCount() function generation.
*/
class PartitionParameterConfig {
  public:
        const char *arrayName;
        int dimensionNo;
};

/* function definition to generate get-partition-count() routine for any given LPS */
List<PartitionParameterConfig*> *generateLPUCountFunction(std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials, Space *space);

/* function that calls the above function repeatedly to generate get-partition-count() functions for all LPSes. */
Hashtable<List<PartitionParameterConfig*>*> *generateLPUCountFunctions(const char *headerFile, 
		const char *programFile, 
		const char *initials, MappingNode *mappingRoot);

/* definition for the function that generate a routine to construct an LPU given its ID */
void generateLpuConstructionFunction(std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials, Space *lps);

/* function that calls the aforementioned function for all LPSes in the task */
void generateAllLpuConstructionFunctions(const char *headerFile,
		const char *programFile, const char *initials, MappingNode *mappingRoot);

#endif
