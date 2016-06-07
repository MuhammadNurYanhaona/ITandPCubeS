#ifndef _H_space_mapping
#define	_H_space_mapping

#include "../utils/list.h"
#include "../semantics/task_space.h"
#include <iostream>

// two preprocessor constants to distinguish between two possible interpretations of a hybrid PCubeS
// architecture
#define PCUBES_MODEL_TYPE_HOST_ONLY 1
#define PCUBES_MODEL_TYPE_HYBRID 2

/* object definition to keep track of the configuration of a PCubeS space */
class PPS_Definition {
  public:
	int id;
	const char *name;
	int units;

	/* We need a variable that designate the PPS representing CPU cores. This is required to 
	   manage thread affinity. For the sake of identification, the current requirement is 
	   that in the PCubeS description file, the space correspond to CPU cores should be marked 
	   as <core> besides its name.
	*/
	bool coreSpace;

	/* We need another variable to identify where memory segmentation occurs in the hardware.
	   That is, either we have distributed memory in the PPS or there is non-uniformity in 
	   the access latency for memory modules. Note that if the memory of a PPS is segmented
	   among its PPUs then any memory available in an ancestor PPS in not directly available
	   in those PPUs either as an addressible memory.
	   The marker '<segment>' after a PPS name indicates that it has segmented memory.	 	
	*/
	bool segmented;

	/* We need a variable to identify where in the PCubeS hierarchy the separation of physical
	   hardware happens. This information is needed to determine when to restart the processor
	   numbering while assigning threads to processors.
	   The marker '<unit>' after a PPS name indicates that in this space physical separation
	   happens between hardware units.
	*/
	bool physicalUnit;

	/* The hybrid model is identified by the presence of a PPS that has '<gpu>' marker next to 
 	   the PPS name. Given that at this momemnt we are targetting NVIDIA GPGPUs only, we know 
	   that there will be exactly two more intra GPU PPSes for SMs and Warps. So just one marker
	   is needed for the top-most gpu PPS		
	*/
	bool gpuTransition;
  public:
	PPS_Definition();
	void print(int indentLevel);
};

class PCubeSModel {
  protected:
	int modelType;
	const char *modelName;
	List<PPS_Definition*> *ppsDefinitions;
  public:
	PCubeSModel(int modelType);
	PCubeSModel(int modelType, List<PPS_Definition*> *ppsDefinitions);
	void setModelName(const char * modelName) { this->modelName = modelName; }
	const char *getModelName() { return modelName; }
	int getModelType() { return modelType; }
	void addNextPpsDefinition(PPS_Definition *ppsDef) { ppsDefinitions->Append(ppsDef); }
	List<PPS_Definition*> *getPpsDefinitions() { return ppsDefinitions; }
	void print(int indentLevel);	

	// these three functions are applicable only for the hybrid PCubeS model	
	int getGpuTransitionSpaceId();
	int getSMCount();
	int getWarpCount();
};

/* object definition to identify an LPS-PPS mapping */
class MapEntry {
  public:
	Space *LPS;
	PPS_Definition *PPS;
};

/* object definition to generate mapping hierarchy from partition and mapping configurations */
class MappingNode {
  public:
	MappingNode *parent;
	MapEntry *mappingConfig;
	int index;
	List<MappingNode*> *children;
};

/* function definition to read the PCubeS description of the hardware from a file */
List<PCubeSModel*> *parsePCubeSDescription(const char *filePath);

/* function defintion to parse the mapping configuration file */
MappingNode *parseMappingConfiguration(const char *taskName, 
		const char *filePath, 
		PartitionHierarchy *lpsHierarchy, 
		List<PCubeSModel*> *pcubesModels);

/* function definition to generate constants corresponds to LPSes */
void generateLPSConstants(const char *outputFile, MappingNode *mappingRoot);

/* function definition to generate the thread counts for all PPSes */
void generatePPSCountConstants(const char *outputFile, PCubeSModel *pcubesModel); 

/* 
   We need to know what processor Id of the target hardware correspond to what actual physical
   unit in the hardware. Otherwise, there may be a mismatch in the expected behavior of the
   code from its actual performance as processor numbering does not necessarily happen in an
   increasing order. This functions parse the processor description file, created by inpecting
   the /proc/cpuinfo file, and generate an array that sort processor Ids so that we can get
   the physical unit intended for a virtual processor id.
*/
void generateProcessorOrderArray(const char *outputFile, const char *processorFile);

#endif
