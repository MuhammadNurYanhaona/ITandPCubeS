#ifndef _H_space_mapping
#define	_H_space_mapping

/* This header file contains classes and utility functions related to parsing and interpreting
   pcubes and mapping configurations from files.
*/

#include "../../../../common-libs/utils/list.h"
#include "../../../../frontend/src/semantics/task_space.h"
#include <iostream>

/* object definition to keep track of the configuration of a PCubeS space (aka a PPS) */
class PPS_Definition {
  public:
	int id;
	const char *name;
	int units;

	/* We need a variable that designate the PPS representing CPU cores. This is required to 
	   manage thread affinity. For the sake of identification, the current requirement is 
	   that the space correspond to CPU cores should be flagged as <core> besides its name 
	   in the PCubeS description file.
	*/
	bool coreSpace;

	/* We need another variable to identify where memory segmentation occurs in the hardware.
	   That is, either we have distributed memory in the PPS or there is non-uniformity in 
	   the access latency for memory modules. Note that if the memory of a PPS is segmented
	   among its PPUs then any memory available in an ancestor PPS in not directly available
	   in those PPUs as a directly addressible memory. The marker '<segment>' after a PPS 
	   name indicates that it has segmented memory.	 	
	*/
	bool segmented;

	/* We need another variable to identify where in the PCubeS hierarchy the separation of 
	   physical hardware happens. This information is needed to determine when to restart 
	   the processor numbering while assigning threads to processors. The flag '<unit>' after 
	   a PPS name indicates that in this space physical separation happens among hardware 
	   units.
	*/
	bool physicalUnit;

	void print(int indentLevel);
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
   We need to know what processor ID of the target hardware correspond to what actual physical
   unit in the hardware. Otherwise, there may be a mismatch in the expected behavior of the
   code from its actual performance as processor numbering does not necessarily happen in an
   increasing order. This function parses the processor description file, created by inpecting
   the /proc/cpuinfo file, and generate an array that sort processor IDs so that we can get
   the physical unit intended for a virtual processor ID.
*/
void generateProcessorOrderArray(const char *outputFile, const char *processorFile);

#endif
