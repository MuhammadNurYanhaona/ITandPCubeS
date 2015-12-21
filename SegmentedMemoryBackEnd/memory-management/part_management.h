#ifndef _H_part_management
#define _H_part_management

/* This header file lists classes that are used to hold the data content of a PPU during a task execution.
   As an LPU is scheduled for execution, instances of this classes are consulted to generate appropriate
   metadata and set proper data references to the template LPU description based on partition configurations.	
   This classes are also used to deduce data interval configurations of other PPUs that the current PPU may
   interact with to synchronize on update of shared data. Note that most of the functionalities offered in
   this library are actually implemented in other libraries of the memory-management module. Therefore, this
   library is just a kind of a convenient interface to manage data at runtime.  
*/

#include "../utils/utility.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "allocation.h"
#include "part_tracking.h"
#include "part_generation.h"

/* This class holds the configuration and content of a data structure of a single LPS  handled by a PPU */
class DataItems {
  protected:
	// name of the data structure
	const char *name;
	// dimensionality of the data structure
	int dimensionality;
	// partition configuration for each dimension
	List<DimPartitionConfig*> *dimConfigList;
	// generated data partition config from individual dimension configuration
	DataPartitionConfig *partitionConfig;
	// structure holding the list of data parts that belong to current PPU 
	DataPartsList *partsList;
	// a flag indicating that the data items have been initialized and ready to be used in computation
	bool ready;
  public:
	DataItems(const char *name, int dimensionality);
	const char *getName() { return name; }
	int getDimensions() { return dimensionality; }
	void addDimPartitionConfig(int dimensionId, DimPartitionConfig *dimConfig);
	void generatePartitionConfig();
	void setPartitionConfig(DataPartitionConfig *partitionConfig);
	DataPartitionConfig *getPartitionConfig();
	void setPartsList(DataPartsList *partsList) { this->partsList = partsList; }
	DataPartsList *getPartsList() { return partsList; }
	DataPart *getDataPart(List<int*> *partIdList, PartIterator *iterator);
	List<DataPart*> *getAllDataParts();
	PartIdContainer *getPartIdContainer() { return partsList->getPartContainer(); }
	bool isEmpty() { return partsList == NULL || partsList->isInvalid(); }
	
	// an iterator should be maintained by each PPU controller for each data structure for efficient 
	// search of data parts by ids.
	PartIterator *createIterator();
};

/* This class holds LPU data parts of all variables correspond to a single LPS */
class LpsContent {
  protected:
	// id of the LPS
	int id;
	// a mapping from variable names to their data parts
	Hashtable<DataItems*> *dataItemsMap;
  public:
	LpsContent(int id);
	inline void addDataItems(const char *varName, DataItems *dataItems) {
		dataItemsMap->Enter(varName, dataItems);
	}
	inline DataItems *getDataItems(const char *varName) { return dataItemsMap->Lookup(varName); }
	void addPartIterators(Hashtable<PartIterator*> *partIteratorMap);
	bool hasValidDataItems();
};

/* This class holds all data structure informations and references regarding different LPSes of a task */
class TaskData {
  protected:
	Hashtable<LpsContent*> *lpsContentMap;
  public:
	TaskData();
	void addLpsContent(const char *lpsId, LpsContent *content);
	DataItems *getDataItemsOfLps(const char *lpsId, const char *varName);		

	// each PPU-controller (currently a thread) within a segment should get its own set of iterators
	// that it will use to efficiently identify data-parts for its LPUs and to avoid unnecessary
	// memory allocation/de-allocation during part generation and identification processes.
	Hashtable<PartIterator*> *generatePartIteratorMap();

	// This tells if the current segment contains data to be used in computations of a particular LPS.
	// If there is no data then there is no thread in the segment that does computation for that LPS.
	bool hasDataForLps(const char *lpsId);
};

#endif
