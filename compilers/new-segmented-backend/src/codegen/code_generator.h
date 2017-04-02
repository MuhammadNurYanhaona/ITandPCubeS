#ifndef _H_code_generation
#define _H_code_generation

#include "../../../common-libs/utils/list.h"

class Definition;
class TaskGlobalScalar;
class TupleDef;
class TaskDef;
class Space;
class MappingNode;
class PPS_Definition;
class EnvironmentLink;
class ReductionMetadata;

/* function definition to import common header files in generated code and write the namespace */
void initializeOutputFiles(const char *headerFile, 
		const char *programFile, 
		const char *initials, TaskDef *taskDef);

/* function definition for generating constants for total number of threads, threads per core,
   threads per segment, processors per hardware unit, etc.  */
void generateThreadCountConstants(const char *outputFile, 
		MappingNode *mappingRoot, List<PPS_Definition*> *pcubesConfig);

/* function definition for generating the runtime library routine that will create thread Ids */
void generateFnForThreadIdsAllocation(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot, 
		List<PPS_Definition*> *pcubesConfig);

/* In a conventional distributed shared memory system, the user most often does not use the entire
 * machine to run his program. In fact, most of the time the portion of the machine he uses does 
 * not form a symmetrical PCubeS hierarchy either. Therefore the thread ID allocation logic of the
 * previous function that assumes symmetry in the machine/sub-machine architecture, if used without
 * any modification, may result in program deadlock or completion with a partial result due to the
 * absense of some assumed to be participating threads. The procedure generated by the following 
 * function is, thereby, used to adjust the PPU counts and thread-IDs based on actual number of 
 * threads found at runtime.   
 */
void generateFnForThreadIdsAdjustment(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		MappingNode *mappingRoot); 


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
void generateLpuDataStructures(const char *outputFile, 
		MappingNode *mappingRoot, 
		List<ReductionMetadata*> *reductionInfos);

/* function definition to generate print functions for LPUs of different LPSes */
void generatePrintFnForLpuDataStructures(const char *initials, 
		const char *outputFile, MappingNode *mappingRoot, 
		List<ReductionMetadata*> *reductionInfos);

/* function definition to generate classes for all tuple definitions found in the source code */
void generateClassesForTuples(const char *filePath, List<Definition*> *tupleDefList);

/* function definition to generate classes for storing task global and thread local variables */
void generateClassesForGlobalScalars(const char *filePath, 
		List<TaskGlobalScalar*> *globalList,
		Space *rootLps);

/* function definition to translate the initialize block of a task if exists */
void generateInitializeFunction(const char *headerFile,
                const char *programFile, const char *initials,
                List<const char*> *envLinkList, TaskDef *taskDef, Space *rootLps);

/* function definition to generate a list of external library linking requirements on a text file */
void generateExternLibraryLinkInfo(const char *linkDescriptionFile);

/* function definition to close the namespace of the header file after all update is done */
void closeNameSpace(const char *headerFile);


#endif
