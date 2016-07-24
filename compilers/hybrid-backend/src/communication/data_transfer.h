#ifndef DATA_TRANSFER_H_
#define DATA_TRANSFER_H_

/* This header file includes classes that are used to move data in-between communication buffers and operating
 * memory data parts. Note that, during writing from a communication buffer to the operating memory, there might
 * be more than one locations of update for a single index within the communication buffer due to the presence
 * of overlapping in data partitions.
 * Instead of the hierarchically organized part-tracking containers creating and maintaining lists of <data part,
 * location of update> pairs; we adopt the policy of passing the specification of transfer along the way as we
 * try to locate the data part holding the location intended to participate in the transfer, and do the update at
 * the end of the recursive process. If the communication buffer index belongs to multiple overlapping data parts
 * then at each point of overlapping the recursive process bifurcates and follows multiple part containers
 * branches at the next level.
 * This strategy of passing a transfer specification and doing recursive updates allows treating reading and
 * writing the same way.
 * */

#include <iostream>
#include <vector>
#include <cstdlib>
#include "part_config.h"
#include "../runtime/structure.h"
#include "../memory-management/allocation.h"
#include "../partition-lib/partition.h"
#include "../utils/list.h"

enum TransferDirection { COMM_BUFFER_TO_DATA_PART, DATA_PART_TO_COMM_BUFFER };

/* This class describes the beginning of the storage index of a particular data point in a data part. The index is
 * later to be used to find the location to read data from or write data to during the interchange between the 
 * operating memory and communucation buffers when the underlying data structure has multiple versions. For version-
 * less data structures, the update location could be recorded, giving a lower transfer cost, directly instead of
 * deriving the location from the index. 
 */
class DataPartIndex {
  protected:
	DataPart *dataPart;
	int index;
  public:
	DataPartIndex() {
		this->dataPart = NULL;
		this->index = -1;
	}
	DataPartIndex(DataPart *dataPart, int index) {
		this->dataPart = dataPart;
		this->index = index;	
	}
	inline char *getLocation() {
		void *data = dataPart->getData();
		char *charData = reinterpret_cast<char*>(data);
		return charData + index;
	}
	inline DataPart *getDataPart() { return dataPart; }
	inline int getIndex() { return index; }
};

/* This class is usefull when more than one location in a data part might need to be accessed (typically for writing 
 * purpose) for an index in the communication buffer. For example, if the data parts have boundary over-lappings in 
 * the form of padding then we may have different data parts in the same segments that should receive an update from 
 * the communication buffer.
 */
class DataPartIndexList {
  protected:
	List<DataPartIndex> *partIndexList;
  public:
	DataPartIndexList() { partIndexList = new List<DataPartIndex>; }
	~DataPartIndexList() { delete partIndexList; }
	inline void addPartIndex(DataPartIndex partIndex) { partIndexList->Append(partIndex); }
	inline List<DataPartIndex> *getPartIndexList() { return partIndexList; }
};

/* class holding all instructions regarding a single data-point transfer between the communication buffer and the
 * operating memory
 * */
class TransferSpec {
  protected:
	// indicates read from or write to the communication buffer location
	TransferDirection direction;
	// indicates the step size to be used to identify the operating memory location and how much data to copy;
	// here the size is specified in terms of the number of characters a single data item is equivalent to
	int elementSize;
	// address in the communication buffer for read/write
	char *bufferEntry;
	// the actual index of the data-point retrieved from the interval representation of the communication buffer
	std::vector<int> *dataIndex;
  public:
	TransferSpec(TransferDirection direction, int elementSize);
	TransferDirection getDirection() { return direction; }
	virtual ~TransferSpec() {}
	void setBufferEntry(char *bufferEntry, std::vector<int> *dataIndex);
	inline std::vector<int> *getDataIndex() { return dataIndex; }
	inline int getStepSize() { return elementSize; }

	// function to be used to do the data transfer once the participating location in the operating memory has been
	// identified
	virtual void performTransfer(DataPartIndex dataPartIndex);
};

/* This subclass of transfer specification is used in the case where we do not intend to do a data transfer at an
 * instance; rather, we intend to get the address within the operating memory where data should be written into or
 * should be read from later.
 * */
class TransferLocationSpec : public TransferSpec {
  private:
	char **bufferLocation;
  public:
	// the transfer direction used here is irrelevant; one is picked because the super class constructor needs one
	TransferLocationSpec(int elementSize) : TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize) {
		bufferLocation = NULL;
	}
	void setBufferLocation(char **bufferLocation) {
		this->bufferLocation = bufferLocation;
	}
	void performTransfer(DataPartIndex dataPartIndex) {
		char *dataPartLocation = dataPartIndex.getLocation();
		*bufferLocation = dataPartLocation;
	}
};

/* This subclass of transfer specification serve the purpose similar to Transfer-Location-Spec class of the above but 
 * for data structures having multiple versions. For those data structures, the memory location of update/read for a 
 * particular point in the communication buffer shifts as different versions occupies separate memory addresses but the
 * index being accessed within those memory allocations does not change.
 *
 * Notice that a list of data-part-index has been maintained instead of just one. This is done because there might be
 * more than one operating memory data part location per entry in the communication buffer.
 */
class TransferIndexSpec : public TransferSpec {
  private:
	DataPartIndexList *partIndexListRef;
  public:
	// the transfer direction used here is irrelevant; one is picked because the super class constructor needs one
	TransferIndexSpec(int elementSize) : TransferSpec(COMM_BUFFER_TO_DATA_PART, elementSize) {
		partIndexListRef = NULL;
	}
	void setPartIndexListReference(DataPartIndexList *indexListRef) {
		this->partIndexListRef = indexListRef;
	}
	void performTransfer(DataPartIndex dataPartIndex) {
		partIndexListRef->addPartIndex(dataPartIndex);
	}
};

/* class holding information that is needed to traverse the part-container hierarchy and identify the location of a data 
 * transfer
 * */
class DataPartSpec {
  private:
	int dimensionality;
	Dimension *dataDimensions;
	List<DataPart*> *partList;
	DataItemConfig *dataConfig;
  public:
	DataPartSpec(List<DataPart*> *partList, DataItemConfig *dataConfig);
	~DataPartSpec() { delete[] dataDimensions; }
	inline DataItemConfig *getConfig() { return dataConfig; }
	int getDimensionality() { return dimensionality; }

	// The identification of the memory address for a transaction with the communication buffer involves applying
	// transformation to the original data index from the buffer at each point of the part-container tree traversal.
	// The process needs to keep track of the current state of the index to proceed to the next step. To avoid
	// creating a new index tracker object for each data transfer, we maintain a single tracker object (represented
	// by the second parameter) and initiate/reset it before starting the tree traversal.
	void initPartTraversalReference(std::vector<int> *dataIndex,
			std::vector<XformedIndexInfo*> *transformVector);

	// function to be used at the end of part-container tree hierarchy traversal to get the memory location that
	// should participate in a data transfer
	char *getUpdateLocation(PartLocator *partLocator, std::vector<int> *partIndex, int dataItemSize);

	// function to be used at the end of part-container tree hierarchy traversal to get the data part index that
	// should participate in a data transfer
	DataPartIndex getDataPartUpdateIndex(PartLocator *partLocator, std::vector<int> *partIndex, int dataItemSize);
};

#endif /* DATA_TRANSFER_H_ */
