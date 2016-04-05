#ifndef _H_data_transceiver
#define _H_data_transceiver

/* This header file includes all the structures and features needed to exchange data among segments to syncrhonize two versions
 * of an environmental data item. Such a need for synchronization arises when an item from one task's environment is assigned
 * to a subsequent task having a different partition configuration for that item, or two versions of that items already extant 
 * and one is discovered to be stale as the other has been modified.
 * Note that in both scenarios it may happen that the stale version can be updated by data locally available in the segment, or
 * data exchanges among segments may be needed. In both cases, the elements of this header file take care of all aspects of the 
 * data transfer. */

#include "environment.h"
#include "../utils/interval.h"
#include "../utils/list.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_tracking.h"

#include <mpi.h>
#include <fstream>

class DataExchange;

/* After undertaking all steps before it can be discovered that data movement is needed due to an environment instruction, the
 * problem gets reduced to transferring data from a particular version reference in the program environment of the underlying 
 * structure to an LPS allocation for it in a going to be executed task. This class bears the information detailing the transfer
 * specification */
class TransferConfig {
  private:
	ProgramEnvironment *progEnv;
	const char *dataItemId;
	ListReferenceKey *sourceVersionKey;
	LpsAllocation *receiver;
	TaskItem *receiverTaskItem;
	int elementSize;
	std::ofstream *logFile;
  public:
	TransferConfig(ProgramEnvironment *progEnv, 
		const char *dataItemId, 
		ListReferenceKey *svk, 
		LpsAllocation *recv, TaskItem *rtk);

	const char *getDataItemId() { return dataItemId; }
	ProgramEnvironment *getProgEnv() { return progEnv; }
	LpsAllocation *getReceiver() { return receiver; }
	TaskItem *getReceiverTaskItem() { return receiverTaskItem; }
	PartsListReference *getSourceReference();
	int getElementSize() { return elementSize; }
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }	

	// retrieves or generates an interval description for the local data need in the target LPS allocation
	List<MultidimensionalIntervalSeq*> *getLocalDestinationFold();
	// creates a version reference for the target LPS allocation's data parts to be used registering a new version for it
	// in the program environment
	ListReferenceAttributes *prepareTargetVersionAttributes();
	// create a key for the would be receiver reference for searching for a possibly already existant version
	const char *generateTargetVersionkey();
};

/* This class determines if there is any need for cross segment communication for the data synchronization for the target LPS
 * allocation's parts List */
class CommunicationReqFinder {
  private:
	List<MultidimensionalIntervalSeq*> *localSourceFold;
	List<MultidimensionalIntervalSeq*> *localTargetFold;
  public:
	CommunicationReqFinder(List<MultidimensionalIntervalSeq*> *sourceFold, 
			List<MultidimensionalIntervalSeq*> *targetFold);
	bool isCrossSegmentCommRequired(std::ofstream &logFile);
};

/* This class collects data parts to segments mapping information once it has been determined that cross segment communication
 * is required to synchronize the target LPS allocations data parts in all segments. */
class SegmentMappingPreparer {
  private:
	List<MultidimensionalIntervalSeq*> *localSegmentContent;
  public:
	SegmentMappingPreparer(List<MultidimensionalIntervalSeq*> *localSegmentContent);
	List<SegmentDataContent*> *shareSegmentsContents(std::ofstream &logFile);
};

/* This class determines if their is a scope for local data transfer from the source parts list to the destination parts list
 * and if YES then does the data transfer */
class LocalTransferrer {
  private:
	DataPartitionConfig *sourceConfig;
	PartIdContainer *sourcePartsTree;
	List<DataPart*> *sourcePartList;
	List<MultidimensionalIntervalSeq*> *sourceFold;
	DataPartitionConfig *targetConfig;
	PartIdContainer *targetPartsTree;
	List<MultidimensionalIntervalSeq*> *targetFold;
	List<DataPart*> *targetPartList;
	int elementSize; 
  public:
	LocalTransferrer(TransferConfig *transferConfig);
	
	DataPartitionConfig *getSourceConfig() { return sourceConfig; }
	List<MultidimensionalIntervalSeq*> *getSourceFold() { return sourceFold; }
	PartIdContainer *getSourcePartsTree() { return sourcePartsTree; }
	List<DataPart*> *getSourcePartList() { return sourcePartList; }	
	
	DataPartitionConfig *getTargetConfig() { return targetConfig; }
	List<MultidimensionalIntervalSeq*> *getTargetFold() { return targetFold; }
	PartIdContainer *getTargetPartsTree() { return targetPartsTree; }
	List<DataPart*> *getTargetPartList() { return targetPartList; }
	 
	void transferData(std::ofstream &logFile);
};

/* This class holds a communication data buffer for a data transfer between the local segment and a remote segment. It also
 * retrieves data from the local source parts list (in case of an outgoing transfer) and populate data into the local target
 * parts list (in case of an incoming transfer) */
class TransferBuffer {
  private:
	char *data;
	int sender;
	int receiver;
	int bufferTag;
	DataExchange *exchange;
	DataPartSpec *partListSpec;
	TransferSpec *transferSpec;
	PartIdContainer *partContainerTree;
  public:
	TransferBuffer(int sender, int receiver, 
			DataExchange *exchange, 
			DataPartSpec *partListSpec, TransferSpec *transferSpec, 
			PartIdContainer *partContainerTree);
	~TransferBuffer();
	void preprocessBuffer(std::ofstream &logFile);
	int getSender() { return sender; }
	int getReceiver() { return receiver; }
	void setBufferTag(int bufferTag) { this->bufferTag = bufferTag; }
	int getBufferTag() { return bufferTag; }
	char *getData() { return data; }
	void postProcessBuffer(std::ofstream &logFile);
	int compareTo(TransferBuffer *other);
	int getSize();
  private:
	void processBuffer(std::ofstream &logFile);
};

/* This class prepare data transfer buffers for both incoming and outgoing communications */
class TransferBuffersPreparer {
  private:
	int elementSize;
	List<MultidimensionalIntervalSeq*> *localSourceContent;
	DataPartSpec *sourcePartsListSpec;
	List<MultidimensionalIntervalSeq*> *localTargetContent;
	DataPartSpec *targetPartsListSpec;
  public:
	TransferBuffersPreparer(int elementSize, 
			List<MultidimensionalIntervalSeq*> *localSourceContent, 
			DataPartSpec *sourcePartsListSpec, 
			List<MultidimensionalIntervalSeq*> *localTargetContent, 
			DataPartSpec *targetPartsListSpec);
	List<TransferBuffer*> *createBuffersForOutgoingTransfers(PartIdContainer *sourceContainer, 
			List<SegmentDataContent*> *targetContentMap,
			std::ofstream &logFile);
	List<TransferBuffer*> *createBuffersForIncomingTransfers(PartIdContainer *targetContainer, 
			List<SegmentDataContent*> *sourceContentMap,
			std::ofstream &logFile);
};

/* This class does the MPI data transfers between buffers of current segment with buffers of segments it interacts with */
class BufferTransferrer {
  private:
	bool sendMode;
	List<TransferBuffer*> *bufferList;
	MPI_Request *transferRequests;
  public:
	BufferTransferrer(bool mode, List<TransferBuffer*> *bufferList);
	~BufferTransferrer();
	void sendDataAsync(std::ofstream &logFile);
	void receiveDataAsync(std::ofstream &logFile);
	void waitForTransferComplete(std::ofstream &logFile);	
};

/* This class handles all aspects of data transfer from a parts list in the environment to one of its alternatives in the
 * LPS allocation of an upcoming task */
class DataTransferManager {
  private:
	TransferConfig *transferConfig;
	
	// Since it takes a few group communications to determine what segment has what data, we should try to reuse this 
        // information as much as possible. Thus these two properties are provided to kep the segment to data mappings
        // that are generated/retrieved during execution of the data transfer manager for latter storing in the program
	// environment. 
        List<SegmentDataContent*> *sourceContentMap;
        List<SegmentDataContent*> *targetContentMap;

	// For the target parts list segment to data mapping, we may sometimes want to calculate the segment fold anew, e.g.,
	// if the same task has been invoked again with a different set of partition arguments. This property tells us what
	// choice to make under a particular calling context
	bool computeTargetFoldAfresh; 

	// a stream for logging events during data transfer
        std::ofstream *logFile;
  public:
	DataTransferManager(TransferConfig *transferConfig, bool computeTargetFoldAfresh);
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }
	List<SegmentDataContent*> *getSourceContentMap() { return sourceContentMap; }
	List<SegmentDataContent*> *getTargetContentMap() { return targetContentMap; }

	void handleTransfer();
};

#endif
