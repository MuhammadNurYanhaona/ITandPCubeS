#ifndef _H_code_generation
#define _H_code_generation

#include "../utils/list.h"

class MappingNode;
class PPS_Definition;
class EnvironmentLink;

/* function definition to import common header files in generated code and write the namespace */
void initializeOutputFiles(const char *headerFile, 
		const char *programFile, const char *initials);

/* function definition for generating constants for total number of threads and threads per core  */
void generateThreadCountConstants(const char *outputFile, 
		MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig);

/* function definition for generating the runtime library routine that will create ThreadIds */
void generateFnForThreadIdsAllocation(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot, 
		List<PPS_Definition*> *pcubesConfig);

/* function definition for generating array metadata and environment links structures for a task */
void generateArrayMetadataAndEnvLinks(const char *outputFile, MappingNode *mappingRoot,
		List<EnvironmentLink*> *envLinks);

/* function definition to generate data structures representing LPUs of different LPSes */
void generateLpuDataStructures(const char *outputFile, MappingNode *mappingRoot);

/* function definition to close the namespace of the header file after all update is done */
void closeNameSpace(const char *headerFile);

#endif
