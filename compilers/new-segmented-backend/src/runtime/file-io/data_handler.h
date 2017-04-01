#ifndef _H_data_handler
#define _H_data_handler

/* this is a library of classes that should be overriden to allow read/write of parts of a data structure of an LPS */

#include "stream.h"

#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/interval.h"
#include "../../../../common-libs/domain-obj/structure.h"
	
/* This is just a helper class to retain hierarchical information about a data part while IO is ongoing. Every element
   here is a list of list because hierarchical information for individual dimensions is stored separately. Information
   made available by this class is needed during part index to file location transformation. 	   
*/
class PartInfo {
  public:
	List<List<Dimension*>*> *partDimensions;
	List<List<int>*> *partCounts;
	List<List<int>*> *partIdList;
	
	// This property holds a mathematical description of the content of the data part that originally belongs to
	// it -- regions that are overlapped paddings originally coming from some other data part does not belong to
	// the current part is excluded in this description. It is only used when padding exclusion is required.
	List<MultidimensionalIntervalSeq*> *contentDescription;
	
	PartInfo() {
		partDimensions = new List<List<Dimension*>*>;
		partCounts = new List<List<int>*>;
		partIdList = new List<List<int>*>;
		contentDescription = NULL;
	}
	~PartInfo() { clear(); }
	void clear();
	void generateContentDescription(DataPartitionConfig *partConfig);

	// returns true if the data/cell index is not from an overlapped padding region of the data part; returns false
	// otherwise
	bool isDataIndexInCorePart(List<int> *dataIndex);
};

/* common base class that embodies routines needed for both reading and writing data parts */
class PartHandler {
  protected:
	const char *fileName;	
	List<DataPart*> *dataParts;
	DataPartitionConfig *partConfig;
	int dataDimensionality;
	Dimension *dataDimensions;
	
	// two supporting variables to keep track of what part is been currently under process and its metadata
	DataPart *currentPart;
	PartInfo *currentPartInfo;
	
	// a supporting variable that is used to represent a file location (actual data index) of a particular element
	// in a data part to avoid creating a new list each time we do a data part to file location transformation
	List<int> *currentDataIndex;
 
	// Since some IT partition functions support boundary overlappings in terms of padding, data parts generated 
	// using them may have regions that does not belong them originally. Particularly when writing content to a
	// file, only the regions owned by the data part under consideration should be copied into the file. Since 
	// it adds some validation overhead when processing each cell of a data part, we have this boolen flag field
	// to indicate when padding exclusion should be bothered with.
	bool needToExcludePadding;
  public:
	PartHandler(DataPartsList *partsList, DataPartitionConfig *partConfig);
	void setFileName(const char *fileName) { this->fileName = fileName; }
	void setNeedToExcludePadding(bool stat) { needToExcludePadding = stat; }
	bool doesNeedToExcludePadding() { return needToExcludePadding; }
	
	// this routine iterates the data section of all parts one-by-one for reading/writing   
	virtual void processParts();
	// function to be called immediately after a new part has been selected for processing and before anything else 
	// has been done with it
	void calculateCurrentPartInfo();
	// functions to be used to identify the memory for the part that will receive/send updates in the I/O process
	void *getCurrentPartData() { return currentPart->getData(); }
	// returns one dimensional update index for an element from its, possibly, multidimensional part index 
	long int getStorageIndex(List<int> *partIndex, Dimension *partDimension);
	// returns the data dimension in a list format
	List<Dimension*> *getDimensionList();

	// Task specific subclasses should provide implementation for this function that is needed to identify the 
	// actual data index in the file for an element within a part. As a partitioning function can reorder the 
	// indexes of an array dimension and, further, there can be multiple reordering applied on a data structure's 
	// memory for a particular LPS; a mechanism is needed to reverse transform an element's part index into its 
	// data index in a file.    
	virtual List<int> *getDataIndex(List<int> *partIndex) = 0;
	
	// this function is provided so that subclasses can use appropriate type while reading/writing data 
	virtual void processElement(List<int> *dataIndex, long int storeIndex, void *partStore) = 0;

	// two functions to be used by subclasses to initialize and destroy any resource that may be created for the
	// reading/writing process, e.g., opening and closing I/O streams.
	virtual void begin() = 0;
	virtual void terminate() = 0;

	// a post processing routine to be implemented by any handler to further manipulate the content of the data part
	// recently being read/written
	virtual void postProcessPart(DataPart *dataPart) {}
  private:
	// a recursive helper routine to aid the processParts() function
	void processPart(Dimension *partDimensions, int currentDimNo, List<int> *partialIndex);
};

/* base class to be extended for the reading process */
class PartReader : public PartHandler {
  public:
	PartReader(DataPartsList *partsList, DataPartitionConfig *partConfig) : PartHandler(partsList, partConfig) {}

	// the process element method just call the virtual read element function; this conversion is done to make it
	// explicit the reading process. Task specific subclasses should implement the readElement() function
	void processElement(List<int> *dataIndex, long int storeIndex, void *partStore) {
		readElement(dataIndex, storeIndex, partStore);
	}
	virtual void readElement(List<int> *dataIndex, long int storeIndex, void *partStore) = 0;	
};

/* base class to be extended for the writing process */
class PartWriter : public PartHandler {
  protected:
	// A writer has an ID so that if some ordering is needed among writers of different segments or some specific
	// operation need to be done by the writer of a specific segment such as initializing a output file with 
	// dimension header that can be done. In conventional case, therefore, the writer Id will be the segment Id of
	// the process using it.
	int writerId;
	// for the same reason there is a count variable that keep tracks of the number of participating writers
	int writersCount;
  public:
	PartWriter(int writerId, DataPartsList *partsList, DataPartitionConfig *partConfig) 
			: PartHandler(partsList, partConfig) {
		this->writerId = writerId;
	}
	void setWritersCount(int writersCount) { this->writersCount = writersCount; }

	// Currently file writing happens sequentially. The first segment writes then notifies the second, then the 
	// second segment writes and notifies the third, and so on. There needs to be signals that indicate when a 
	// segment can start writing and when it is done. To send/receive these signals, part-writer overrides the
	// base clase function.
	void processParts();

	// like the PartReader, this class also override the processElement() method to make writing process explicit
	// for task specific subclasses  
	void processElement(List<int> *dataIndex, long int storeIndex, void *partStore) {
		writeElement(dataIndex, storeIndex, partStore);
	}
	virtual void writeElement(List<int> *dataIndex, long int storeIndex, void *partStore) = 0;	
};

#endif
