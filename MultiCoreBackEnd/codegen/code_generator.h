#ifndef _H_code_generation
#define _H_code_generation

#include "../utils/list.h"

class TaskGlobalScalar;
class TupleDef;
class TaskDef;
class Space;
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
List<const char*> *generateArrayMetadataAndEnvLinks(const char *outputFile, 
		MappingNode *mappingRoot,
		List<EnvironmentLink*> *envLinks);

/* function definition for generating print routines for envLinks and array metadata objects; it 
   also generate the constructor for array-metadata */
void generateFnForMetadataAndEnvLinks(const char *taskName, const char *initials, 
		const char *outputFile, 
		MappingNode *mappingRoot, 
		List<const char*> *externalLinks);

/* function definition to generate data structures representing LPUs of different LPSes */
void generateLpuDataStructures(const char *outputFile, MappingNode *mappingRoot);

/* function definition to generate print functions for LPUs of different LPSes */
void generatePrintFnForLpuDataStructures(const char *initials, 
		const char *outputFile, MappingNode *mappingRoot);

/* function definition to generate a routine to initialze the content of the root LPS */
void generateFnToInitiateRootLPSContent(const char *headerFile, 
		const char *programFile, 
		const char *initials,
		TaskDef *taskDef,
		MappingNode *mappingRoot, 
		List<const char*> *externalEnvLinks);

/* function definition to generate a routine to initialize the content of different LPSes except 
   the Root LPS content */
void generateFnToInitiateLPSesContent(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot);

/* This is a simpler version of the preceeding function. It does not allocate any memory for structures
   in any LPS; rather it just makes all other LPS structure references to point to the references of
   that of the root LPS. 
*/
void generateFnToInitiateLPSesContentSimple(const char *headerFileName, 
		const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot);

/* function definition to close the namespace of the header file after all update is done */
void closeNameSpace(const char *headerFile);

/* function definition to generate classes for all tuple definitions found in the source code */
void generateClassesForTuples(const char *filePath, List<TupleDef*> *tupleDefList);

/* function definition to generate classes for storing task global and thread local variables */
void generateClassesForGlobalScalars(const char *filePath, List<TaskGlobalScalar*> *globalList);

/* function definition to translate the initialize block of a task if exists */
void generateInitializeFunction(const char *headerFile, 
		const char *programFile, const char *initials, 
		List<const char*> *envLinkList, TaskDef *taskDef, Space *rootLps);	

#endif
