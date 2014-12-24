#ifndef _H_code_generation
#define _H_code_generation

#include "../utils/list.h"

class MappingNode;
class PPS_Definition;

/* function definition to import common header files in the generated code */
void initializeOutputFile(const char *filePath);

/* function definition for generating macros for total number of threads and threads per core  */
void generateThreadCountMacros(const char *outputFile, 
		MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig);

/* function definition for generating the runtime library routine that will create ThreadIds */
void generateFnForThreadIdsAllocation(const char *outputFile, 
		MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig);

#endif
