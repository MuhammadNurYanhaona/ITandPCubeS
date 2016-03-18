#ifndef _H_allocation
#define _H_allocation

/* This header file specifies all classes related to handling of operating memory of different data structures
   within an IT code (by operating memory, we understand the actual memory locations where access and updates
   take place during the execution of the IT program). We store parts of different data structures as they are
   needed by a particular segmented-PPU. In other word, the data is owned by the PPU and LPUs are constructed
   by passing them the appropriate part references to different data structure. This approach is used to take
   into account that many LPUs may share the same piece of data due to replication and we should avoid storing
   same data piece multiple times. 
*/

#include "part_tracking.h"
#include "part_generation.h"
#include "../utils/list.h"
#include "../utils/utility.h"
#include "../runtime/structure.h"
#include <vector>

/* This class holds all information to identify a part of a data structure configured for a particular LPS and
   to determine how to access/manipulate its content appropriately
*/
class PartMetadata {
  protected:
	// the number of dimensions in the data structure
	int dimensionality;
	// possibly multi-dimensional part Id specifying the index of the part within the whole; note that we
	// have a list here as partitioning in IT is hierarchical and for identification of a part in an LPS we 
	// may need to identify its ancestor parts in higher LPSes
	List<int*> *idList;
	// spread of the part along different dimensions
	// note that even for data reordering partition functions we should have a contiguous spread for a data 
	// part as then we will consider all indexes within the data has been transformed in a way to conform 
	// with the reordering  
	Dimension *boundary;
	// if padding is used at partitioning then its overlapping along of boundaries with neighboring parts
	int *padding;
  public:
	PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary, int *padding);
	~PartMetadata();
	int getSize();
	inline int getDimensions() { return dimensionality; }
	inline List<int*> *getIdList() { return idList; }
	inline Dimension *getBoundary() { return boundary; }
	inline int *getPadding() { return padding; }
	bool isMatchingId(List<int*> *candidateId);
	
	// this is the method to be used to populate the storage dimension information properly for a data part
	// within an LPU
	void updateStorageDimension(PartDimension *partDimension);
};

/* This is an extension to the SuperPart construction within part-tracking library for linking list of data part
   allocations with their faster searching mechanism. This class can be avoided if the data part-list directly
   operates on the part container instead of using it as a mechanism to determine what part to return. We adopt
   this current strategy as the list based part allocation mechanism was developed earlier and the efficient search 
   mechanism has been developed after we faced severe performance and memory problems in straightforward process.
   We did not want to change all the dependencies of the data-part-list construct; rather we only wanted to 
   eliminate those problems.
*/
class PartLocator : public SuperPart {
  protected:
	int partListIndex;
  public:
	PartLocator(List<int*> *partId, int dataDimensions, int partListIndex) 
			: SuperPart(partId, dataDimensions) {
		this->partListIndex = partListIndex;
	}
	inline int getPartListIndex() { return partListIndex; }
};

/* This class holds the metadata and actual memory allocation for a part of a data structure */
class DataPart {
  protected:
	// a reference to part dimensions and other metadata information 
	PartMetadata *metadata;
	// Epoch dependent data structures will have multiple copies, one per epoch, of each part stored within a 
	// PPU. So the epoch count is needed, which is by default set to 1.
	int epochCount;
	// a variable to keep track of the head of the circular array 
	int epochHead;
	// a circular array of allocation units, one for each epoch version
	std::vector<void*> *dataVersions;
	// size of each element of the data part in terms of the number of characters
	int elementSize;
  public:
	DataPart(PartMetadata *metadata, int epochCount, int elementSize);
	~DataPart();

	// because this is a templated function, its implementation needs to be in the header file
	template <class type> static void allocate(DataPart *dataPart) {
		int size = dataPart->metadata->getSize();
		Assert(size > 0);
		int versionCount = dataPart->epochCount;
		std::vector<void*> *dataVersions = dataPart->dataVersions;
		for (int i = 0; i < versionCount; i++) {
			void *data = new type[size];
			Assert(data != NULL);
			char *charData = reinterpret_cast<char*>(data);
			int charSize = size * dataPart->elementSize;
			for (int i = 0; i < charSize; i++) {
				charData[i] = 0;
			}
			dataVersions->push_back(data);
		}
	}

	inline PartMetadata *getMetadata() { return metadata; }

	// returns the memory reference of the allocation unit at the current epoch-head
	void *getData();
	// returns the memory reference of the allocation unit for a specific epoch version
	void *getData(int epoch);
	// moves the head of the circular array one step ahead
        inline void advanceEpoch() { epochHead = (epochHead + 1) % epochCount; }

	// This function is used by multi-versioned data parts to copy values from one allocation to all other
	// allocations. This operation is typically needed when the data part is read from some external file.
	// The contract for multi-versioned data parts is that initially, i.e. before the task starts execution, 
	// all versions have the content. 
	void synchronizeAllVersions();

	// These two functions are added to support data part allocation and content-copying from the environment
	// management module. 
	// Note that cloning of content from a data part in the program environment to a part in a task's 
	// execution environment will be taken place only when both data-parts are expected to contain the same 
	// regions of the underlying data structure.
	void clone(DataPart *other);
	// The location in the environment management module where data part allocation may take place does not
	// have content's data type information. Therefore an allocation mechanism is needed that is independent 
	// of the element type. The default -1 value of the parameter means that all versions of should be 
	// allocated. Otherwise, only those versions that are on or above the threshold should be allocated
	void allocate(int versionThreshold = 0);	
};

/* This class provides generic information about all the parts of an LPS data structure that a segment holds */
class ListMetadata {
  protected:
	// dimensionality of the data structure
	int dimensionality;
	// spread of the data structure along different dimensions
	Dimension *boundary;
	// If the partitioning of a data structure involves padding then parts within a single PPU may need to be 
	// synchronized for overlappings in their boundary regions. The following flag indicates if that is needed 
	// for the current data structure
	bool hasPadding;
  public:
	ListMetadata(int dimensionality, Dimension *boundary);
	inline int getDimensions() { return dimensionality; }
	inline Dimension *getBoundary() { return boundary; }
	inline void setPadding(bool hasPadding) { this->hasPadding = hasPadding; }
	inline bool isPadded() { return hasPadding; }
};

/* This is the class holding all parts of a particular data structures that a PPU holds for some LPS. Note that, a 
   new data parts list should be created within a PPU for a data if there is none already, there is a reordering 
   of data since the last LPS configuration been used for allocation, or a new LPS using the data is found along a 
   path in the partition hierarchy that is not related to any other LPSes the data has been allocated for. Any LPS 
   that does not use the data in any computation should maintain a reference to the list of its nearest decendent 
   or ancestor LPS.  
*/
class DataPartsList {
  protected:
	// a reference to the data structure dimensions and other metadata information
	ListMetadata *metadata;	  
	// part-id-tracking container to be used for quick identification of data parts by part-ids
	PartIdContainer *partContainer;
	// a circular array of data-part-list; there is one list per epoch 
	List<DataPart*> *partList;
	// a tracking variable for determining the number of epochs each part of this list has
	int epochCount;
	// a flag to indicate that this part list is empty and acting as a placeholder only
	bool invalid;
  public:
	DataPartsList(ListMetadata *metadata, int epochCount);
	~DataPartsList();
	
	void initializePartsList(DataPartitionConfig *partConfig, 
			PartIdContainer *partContainer, 
			int partElementSize);

	// because this is a templated function, its implementation needs to be in the header file
	template <class type> static void allocateParts(DataPartsList *dataPartsList) {
		if (dataPartsList->invalid) return;
		List<DataPart*> *parts = dataPartsList->partList;
		for (int i = 0; i < parts->NumElements(); i++) {
			DataPart *dataPart = parts->Nth(i);
			DataPart::allocate<type>(dataPart);
		}
	} 
	
	inline ListMetadata *getMetadata() { return metadata; }
	inline PartIdContainer *getPartContainer() { return partContainer; }
	inline int getEpochCount() { return epochCount; }
	inline List<DataPart*> *getPartList() { return partList; }
	inline bool isInvalid() { return invalid; }
	DataPart *getPart(List<int*> *partId, PartIterator *iterator);

	// each PPU-controller within a segment should get an iterator for each data part list that to be used later 
	// for part searching
	PartIterator *createIterator();
};

#endif
