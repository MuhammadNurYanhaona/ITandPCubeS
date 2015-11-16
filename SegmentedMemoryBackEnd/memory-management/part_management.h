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
	// the number of epoch step needs to be retained if the structure is epoch dependent
	int epochCount;
	// a flag indicating that the data items have been initialized and ready to be used in computation
	bool ready;
  public:
	DataItems(const char *name, int dimensionality, int epochCount);
	const char *getName() { return name; }
	int getDimensions() { return dimensionality; }
	void addDimPartitionConfig(int dimensionId, DimPartitionConfig *dimConfig);
	void generatePartitionConfig();
	void setPartitionConfig(DataPartitionConfig *partitionConfig);
	DataPartitionConfig *getPartitionConfig();
	void setPartsList(DataPartsList *partsList) { this->partsList = partsList; }
	DataPartsList *getPartsList() { return partsList; }
	// an iterator should be maintained by each PPU controller for each data structure for efficient 
	// search of data parts by ids.
	PartIterator *createIterator();
	// function to get the most uptodate version of a part of the structure
	DataPart *getDataPart(List<int*> *partIdList, PartIterator *iterator);
	// function to get an older epoch version of a part
	DataPart *getDataPart(List<int*> *partidList, int epoch, PartIterator *iterator);
	List<DataPart*> *getAllDataParts();
	PartIdContainer *getPartIdContainer() { return partsList->getPartContainer(); }
	bool isEmpty() { return partsList == NULL || partsList->isInvalid(); }
	virtual void advanceEpoch() { partsList->advanceEpoch(); }
};

/* Scalar variables are dimensionless; therefore do not mesh well with the DataItems class structure. 
   Regardless, we want to make an uniform interface for epoch dependency and holding LPU contents. Thereby,
   this class has been added to extend the DataItems class.
*/
class ScalarDataItems : public DataItems {
  protected:
	// to be generic, the versions of the scalar variable are stored as void pointers; a circular array
	// of these pointers are maintained for version dependency
	void **variableList;
	// points to the most recent version of the 
	int epochHead;
  public:
	ScalarDataItems(const char *name, int epochCount);
	template <class type> static void allocate(ScalarDataItems *items) {
		Assert(items->epochCount > 0);
		items->variableList = (void **) new type*[items->epochCount];
		Assert(items->variableList != NULL);
		int size = sizeof(type) / sizeof(char);
		Assert(size > 0);
		for (int i = 0; i < items->epochCount; i++) {
			type *version = new type;
			Assert(version != NULL);
			items->variableList[i] = (void*) version;
			// zero initialize the version
			char *charVersion = reinterpret_cast<char*>(version);
			for (int i = 0; i < size; i++) {
				charVersion[i] = 0;
			}
		}
		items->ready = true;
	}
	// function to get the reference of the latest version of the variable
	void *getVariable();
	// function to get the reference of some earlier epoch version of the variable
	void *getVariable(int version);
	inline void advanceEpoch() { epochHead = (epochHead + 1) % epochCount; };
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
	void advanceItemEpoch(const char *varName);
	void addPartIterators(Hashtable<PartIterator*> *partIteratorMap);
};

/* This class holds all data structure informations and references regarding different LPSes of a task */
class TaskData {
  protected:
	Hashtable<LpsContent*> *lpsContentMap;
  public:
	TaskData();
	void addLpsContent(const char *lpsId, LpsContent *content);
	DataItems *getDataItemsOfLps(const char *lpsId, const char *varName);		
	void advanceItemEpoch(const char *lpsId, const char *varName);

	// each PPU-controller (currently a thread) within a segment should get its own set of iterators
	// that it will use to efficiently identify data-parts for its LPUs and to avoid unnecessary
	// memory allocation/de-allocation during part generation and identification processes.
	Hashtable<PartIterator*> *generatePartIteratorMap();
};

#endif
