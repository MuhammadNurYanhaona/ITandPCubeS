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

#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../codegen/structure.h"

/* This class holds all information to identify a part of a data structure configured for a particular LPS and
   to determine how to access/manipulate its content appropriately
*/
class PartMetadata {
  protected:
	// the number of dimensions in the data structure
	int dimensionality;
	// possibly multi-dimensional part Id specifying the index of the part within the whole; note that we
	// have a list here as partitioning in IT is hierarchical and for identificant of a part in an LPS we 
	// may need to identify its ancestor parts in higher LPSes
	List<int*> *idList;
	// spread of the part along different dimensions
	// note that even for data reordering partition functions we should have a contiguous spread for a data 
	// part as then we will consider all indexes within the data has been transformed in a way to conform 
	// with the reordering  
	Dimension *boundary;
	// if padding is used at partitioning then its overlappings along of boundaries with neighboring parts
	int *padding;
	// the interval description for the part with and without padding
	HyperplaneInterval *coreInterval;
	// this is valid for only those partition functions that support paddings
	HyperplaneInterval *paddedInterval;
  public:
	PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary, int *padding);
	void setIntervals(HyperplaneInterval *coreInterval, HyperplaneInterval *paddedInterval);
	int getSize();
	inline int getDimensions() { return dimensionality; }
	inline List<int*> *getIdList() { return idList; }
	inline Dimension *getBoundary() { return boundary; }
	inline int *getPadding() { return padding; }
	inline HyperplaneInterval *getCoreInterval() { return coreInterval; }
	inline HyperplaneInterval *getPaddedInterval() { return paddedInterval; }
	bool isMatchingId(List<int*> *candidateId);
};

/* This class holds the metadata and actual memory allocation for a part of a data structure
*/
class DataPart {
  protected:
	PartMetadata *metadata;
	void *data;
  public:
	DataPart(PartMetadata *metadata) { 
		this->metadata = metadata;
		this->data = NULL; 
	}
	template <class type> static void allocate(DataPart *dataPart) {
		int size = dataPart->metadata->getSize();
		dataPart->data = new type[size];
	}
	inline PartMetadata *getMetadata() { return metadata; }
	void *getData() { return data; }	
};

/* This class describes all the parts of a data structure that a PPU hold. This metadata is needed to derive infor-
   mation about communication buffers for PPU-PPU interactions. It is, however, mostly needed during constructions 
   of LPUs for computation-stage executions and determining how to fill in data recieved from others and to put 
   data to other buffers when sending to and from individual data parts.
*/
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
	// interval specification for the union of all parts within the list; this ignores padding
	IntervalSet *intervalSpec;
	// interval specification that includes padding within data parts
	IntervalSet *paddedIntervalSpec;
  public:
	ListMetadata(int dimensionality, Dimension *boundary);
	inline int getDimensions() { return dimensionality; }
	inline Dimension *getBoundary() { return boundary; }
	inline void setPadding(bool hasPadding) { this->hasPadding = hasPadding; }
	inline bool isPadded() { return hasPadding; }
	void generateIntervalSpec(List<PartMetadata*> *partList);
	inline IntervalSet *getCoreIntervalSpec() { return intervalSpec; }
	inline IntervalSet *getPaddedIntervalSpec() { return paddedIntervalSpec; }
};

/* This is the class holding all parts of a particular data structures that a PPU holds for some LPS. Although it
   is called a list, an instance can hold multiple lists for epoch dependent data structure where there will be 
   one version of the list for each epoch. Note that, a new data parts list should be created within a PPU for a
   data if there is none already, there is a reordering of data since the last LPS configuration been used for 
   allocation, or a new LPS using the data is found along a path in the partition hierarchy that is not related
   to any other LPSes the data has been allocated for. Any LPS that does not use the data in any computation 
   should maintain a reference to the list of its nearest decendent or ancestor LPS.  
*/
class DataPartsList {
  protected:
	ListMetadata *metadata;	  
	// Epoch dependent data structures will have multiple copies, one per epoch, of each part stored within a 
	// PPU. So the epoch count is needed, which is by default set to 1.
	int epochCount;
	// a circular array of data-part-list; there is one list per epoch 
	List<DataPart*> **partLists;
	// a variable to keep track of the head of the circular array 
	int epochHead;
  public:
	DataPartsList(ListMetadata *metadata, int epochCount);
	template <class type> static void allocate(DataPartsList *dataPartsList, 
			List<PartMetadata*> *partMetadataList) {
		for (int i = 0; i < partMetadataList->NumElements(); i++) {
			PartMetadata *partMetadata = partMetadataList->Nth(i);
			for (int t = 0; t < dataPartsList->epochCount; t++) {
				dataPartsList->partLists[t]->Append(new DataPart(partMetadata));
				DataPart::allocate<type>(dataPartsList->partLists[t]->Nth(i));
			}
		}
	}
	DataPart *getPart(List<int*> *partId);
	DataPart *getPart(List<int*> *partId, int epoch);	
	// moves the head of the circular array one step ahead
	inline void advanceEpoch() { epochHead = (epochHead + 1) % epochCount; }
	inline int getEpochCount() { return epochCount; }
	inline List<DataPart*> *getCurrentList() { return partLists[epochHead]; }
};

#endif
