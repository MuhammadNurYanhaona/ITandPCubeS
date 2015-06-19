#ifndef _H_data_handler
#define _H_data_handler

/* this is a library of classes that should be overriden to allow read/write of parts of a data structure of an LPS */

#include "stream.h"
#include "../memory-management/allocation.h"
#include "../utils/list.h"
#include "../codegen/structure.h"

/* common base class that embodies routines needed for both reading and writing data parts */
class PartHandler {
  protected:
	const char *fileName;	
	List<DataPart*> *dataParts;
	DataPart *currentPart;
	int dataDimensionality;
	Dimension *dataDimensions;
  public:
	PartHandler(DataPartsList *partsList, const char *fileName);
	
	// this routine iterates the data section of all parts one-by-one for reading/writing   
	void processParts();
	// functions to be used to identify the memory for the part that will receive/send updates in the I/O process
	void *getCurrentPartData() { return currentPart->getData(); }
	// returns one dimensional update index for an element from its, possibly, multidimensional part index 
	int getStorageIndex(List<int> *partIndex, Dimension *partDimension);
	// returns the data dimension in a list format
	List<Dimension*> *getDimensionList();

	// Task specific subclasses should provide implementation for this function that is needed to identify the 
	// actual data index in the file for an element within a part. As a partitioning function can reorder the 
	// indexes of an array dimension and, further, there can be multiple reordering applied on a data structure's 
	// memory for a particular LPS; a mechanism is needed to reverse transform an element's part index into its 
	// data index in a file.    
	virtual List<int> *getDataIndex(List<int> *partIndex) = 0;
	
	// this function is provided so that subclasses can use appropriate type while reading/writing data 
	virtual void processElement(List<int> *dataIndex, int storeIndex, void *partStore) = 0;

	// two functions to be used by subclasses to initialize and destroy any resource that may be created for the
	// reading/writing process, e.g., opening and closing I/O streams.
	virtual void begin() = 0;
	virtual void terminate() = 0;
  private:
	// a recursive helper routine to aid the processParts() function
	void processPart(Dimension *partDimensions, int currentDimNo, List<int> *partialIndex);
};

/* base class to be extended for the reading process */
class PartReader : public PartHandler {
  public:
	PartReader(DataPartsList *partsList, const char *fileName) : PartHandler(partsList, fileName) {}

	// the process element method just call the virtual read element function; this conversion is done to make it
	// explicit the reading process. Task specific subclasses should implement the readElement() function
	void processElement(List<int> *dataIndex, int storeIndex, void *partStore) {
		readElement(dataIndex, storeIndex, partStore);
	}
	virtual void readElement(List<int> *dataIndex, int storeIndex, void *partStore) = 0;	
};

/* base class to be extended for the writing process */
class PartWriter : public PartHandler {
  protected:
	// A writer has an id so that if some ordering is needed among writers of different segments or some specific
	// operation need to be done by the writer of a specific segment such as initializing a output file with 
	// dimension header that can be done. In conventional case, therefore, the writer Id will be the segment Id of
	// the process using it 
	int writerId;
  public:
	PartWriter(int writerId, DataPartsList *partsList, const char *fileName) : PartHandler(partsList, fileName) {
		this->writerId = writerId;
	}

	// like the PartReader, this class also override the processElement() method to make writing process explicit
	// for task specific subclasses  
	void processElement(List<int> *dataIndex, int storeIndex, void *partStore) {
		writeElement(dataIndex, storeIndex, partStore);
	}
	virtual void writeElement(List<int> *dataIndex, int storeIndex, void *partStore) = 0;	
};

#endif
