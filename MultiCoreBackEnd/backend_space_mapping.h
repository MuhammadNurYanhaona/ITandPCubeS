#ifndef _H_backend_space_mapping
#define	_H_backend_space_mapping

#include "list.h"
#include "task_space.h"

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

/* function definition for generating the runtime library routine that will create ThreadIds */
void generateFnForThreadIdsAllocation(char *outputFile, MappingNode *mappingRoot);

#endif
