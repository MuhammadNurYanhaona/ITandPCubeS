#ifndef _H_comm_buffer
#define _H_comm_buffer

/* This header file hosts different kinds of communication buffers that are needed in different data synchronization
 * scenarios involving some form of data movements. For each data synchronization instance, there will be one or more
 * communication buffers. Sometimes, there will be more than one types of buffer for a single synchronization. The
 * generic idea is that a segment will create and maintain a list of communication buffer for each synchronization at
 * the beginning of a task launch. As it gets halted on a synchronization at execution time, it will execute read and
 * or write operation on the communication buffers set for that based on what deemed appropriate.
 * Note that this logic does not work for dynamic LPSes but the communication buffers here can be extended to support
 * dynamic LPSes too. Or a new set of buffer can be created for dynamic LPSes each time their data content changes.
 * Platform specific sub-classes should extend these buffers ,e.g, MPI will need extensions for MPI communications
 * */

#include "confinement_mgmt.h"
#include "part_config.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_tracking.h"
#include "../utils/list.h"
#include "../runtime/comm_barrier.h"
#include <vector>

/* To configure the communication buffers properly, we need the data-parts-list representing the operating memory for
 * a data structure for involved LPSes and data item size along with the other information provided by a confinement
 * construction configuration. Thus, this class has been provided.
 * */
class SyncConfig {
  private:
	ConfinementConstructionConfig *confinementConfig;
	DataPartsList *senderDataParts;
	DataPartsList *receiverDataParts;
	int elementSize;
  public:
	SyncConfig(ConfinementConstructionConfig *cc, DataPartsList *sDP, DataPartsList *rDP, int eS) {
		confinementConfig = cc;
		senderDataParts = sDP;
		receiverDataParts = rDP;
		elementSize = eS;
	}
	ConfinementConstructionConfig *getConfinementConfig() { return confinementConfig; }
	DataPartsList *getSenderDataParts() { return senderDataParts; }
	DataPartsList *getReceiverDataParts() { return receiverDataParts; }
	int getElementSize() { return elementSize; }
};

/* The base class for a communication buffer for a data exchange; remember that a data exchange is the configuration
 * of data need to be exchanges between two participating branch of a confinement for the sake of a synchronization.
 * Check the confinement_mgmt.h library for more detail.
 * */
class CommBuffer {
  protected:
	// the exchange this communication buffer has been created for
	DataExchange *dataExchange;
	// data dimension information is needed to access data items from specific indexes on a data-part
	int dataDimensions;
	// the number of data items this communication buffer exchanges
	int elementCount;
	// size of individual data items needed for determining communication buffer size and accessing elements
	int elementSize;
	// identifier for the current segment
	int localSegmentTag;

	// the hierarchical part container trees are needed to quickly identify the data part holding one or more data
	// items included in the communication buffer
	PartIdContainer *senderTree;
	PartIdContainer *receiverTree;

	// the list of operating memory data parts which data items will be read from and/or be written to
	DataPartsList *senderPartList;
	DataPartsList *receiverPartList;

	// data partition configurations for data-parts on two sides of the synchronization are needed to locate the
	// right part and the right location for update within it
	DataItemConfig *senderDataConfig;
	DataItemConfig *receiverDataConfig;
	
	// a buffer identifier to be used as tag for communications if needed
	int bufferTag;
  public:
	CommBuffer(DataExchange *exchange, SyncConfig *syncConfig);
	virtual ~CommBuffer() {}
	DataExchange *getExchange() { return dataExchange; }
	int getBufferSize() { return elementCount * elementSize; }
	int compareTo(CommBuffer *other, bool forReceive);
	int getBufferTag() { return bufferTag; }
	
	// should be overriden by buffer types that has a physical storage for holding data; the default is to through a 
	// fault
	virtual char *getData();
	virtual void setData(char *data);

	// Each subclass should provide its implementation for the following two functions. During the execution of the
	// program, if the computation halts in any synchronization involving communication, the segment controller will
	// get the communication buffer list for the synchronization and invoke read-data or write-data in each of them
	// depending on what is appropriate in that situation. After that the segment controller will resume the threads
	// doing computation.
	virtual void readData() = 0;
	virtual void writeData() = 0;
	
	// these two functions tell if the current segment is sending data, receiving data, or both
	bool isSendActivated();
	bool isReceiveActivated();

	// function to setup the buffer tag
	void setBufferTag(int prefix, int digitsForSegment);
  protected:
	ExchangeIterator *getIterator() { return new ExchangeIterator(dataExchange); }
};

/* This extension is added to shorten the time for data transfer between the operating memory data parts and the
 * communication buffer. Note the algorithm for identifying the location within the operating memory corresponding to a 
 * data item in the data exchange need to do logarithmic searches in the part hierarchy for individual data items. A 
 * simple optimization could be to save the memory locations corresponding to data items and doing the update through 
 * those location pointers. Then the update should take time proportional to the size of the buffer only. A 
 * PreprocessedCommBuffer retrieves and store the memory locations at the beginning for later access.
 *
 * Note that this extension cannot be used for dynamic LPSes.
 * Further, in its current form, this cannot be used for synchronizations involving a data structure having multiple
 * versions either. For that to be supported, we would need memory mappings for each version.
 * */
class PreprocessedCommBuffer : public CommBuffer {
  protected:
	char **senderTransferMapping;
	char **receiverTransferMapping;
  public:
	PreprocessedCommBuffer(DataExchange *exchange, SyncConfig *syncConfig);
	~PreprocessedCommBuffer();
	virtual void readData() = 0;
	virtual void writeData() = 0;
  private:
	// a helper function to traverse a part container tree and get all memory locations for data items that are part
	// of the data-exchange a communication-buffer has been created for
	void setupMappingBuffer(char **buffer,
			DataPartsList *dataPartList,
			PartIdContainer *partContainerTree,
			DataItemConfig *dataConfig);
};

/* Implementation class where there is actually a physical communication buffer that will hold data before a send to 
 * and/or after a receive from of data belonging to the operating memory data parts
 * */
class PhysicalCommBuffer : public CommBuffer {
  protected:
	char *data;
  public:
	PhysicalCommBuffer(DataExchange *exchange, SyncConfig *syncConfig);
	~PhysicalCommBuffer() { delete[] data; }
	void readData();
	void writeData();
	void setData(char *data) { this->data = data; }
	char *getData() { return data; }
  protected:
	void transferData(TransferSpec *transferSpec,
			DataPartSpec *dataPartSpec,
			PartIdContainer *partTree);
};

/* The extension of physical communication buffer to be used with pre-processing enabled
 * */
class PreprocessedPhysicalCommBuffer : public PreprocessedCommBuffer {
  protected:
	char *data;
  public:
	PreprocessedPhysicalCommBuffer(DataExchange *exchange, SyncConfig *syncConfig);
	~PreprocessedPhysicalCommBuffer() { delete[] data; }
	void readData();
	void writeData();
	void setData(char *data) { this->data = data; }
	char *getData() { return data; }
};

/* This extension is for situations where we do not want any intervening memory to be allocated for the communication
 * buffer but, rather, want to copy data directly from one operating memory to another. Obviously, this can only be
 * done when the sender and receiver are both local to the current segment.
 * */
class VirtualCommBuffer : public CommBuffer {
  public:
	VirtualCommBuffer(DataExchange *exchange, SyncConfig *syncConfig) : CommBuffer(exchange, syncConfig) {}
	void readData();
	// Read-write is short-circuited in a virtual buffer. So we need to implement one of the two transfer functions. 
	// To clarify, the segment will call eventually writeData() sometimes after readData() but the act of reading 
	// involves reading into the destination operating memory. So the call for writing is unnecessary.
	void writeData() {}
};

/* This is the virtual communication buffer extension with pre-processing enabled
 * */
class PreprocessedVirtualCommBuffer : public PreprocessedCommBuffer {
  public:
	PreprocessedVirtualCommBuffer(DataExchange *exchange,
			SyncConfig *syncConfig) : PreprocessedCommBuffer(exchange, syncConfig) {}
	void readData();
	void writeData() {}
};

/* This class contains all communication buffers related to a particular data synchronization and does the buffer read
   write as part of the communication. Subclasses should provide implementation for the send() and receive() functions
   to do the actual data transfer.
*/
class CommBufferManager {
  protected:
	// the name of the dependency arc all communications are issued for 
	const char *dependencyName;
	// list of buffers that will be exchanged for the dependency resolution
	List<CommBuffer*> *commBufferList;
  public:
	CommBufferManager(const char *dependencyName);
	~CommBufferManager();
	void setCommBufferList(List<CommBuffer*> *commBufferList) { this->commBufferList = commBufferList; }
	void addCommBuffer(CommBuffer *buffer) { commBufferList->Append(buffer); }
	const char *getName() { return dependencyName; }

	// two functions to pre and post process communication buffers before a send and after a receive respectively
	// these basically read and write the communication buffers
	virtual void prepareBuffersForSend();
	virtual void processBuffersAfterReceive();	

	// functions to be implemented by the sync-type specific subclasses for send and receive; the signal type 
	// indicates if the invoking PPU is interested in communication or not and the iteration represents the invo-
	// cation count from the PPU for any synchronization that happens repeatedly in the course of execution  
	virtual void send(SignalType signal, int iteration) = 0;
	virtual void receive(SignalType signal, int iteration) = 0;
	
  protected:
	// Communication buffers are sorted before send/receive to reduce the data buffering time in segments/processes
	// the sorting here is dependent on what particular role the executing segment is going to play at the present
	// instance. If no second parameter has been provided then the sorting is done over all comm buffers of this 
	// buffer manager; otherwise the argument list gets sorted
	List<CommBuffer*> *getSortedList(bool sortForReceive, List<CommBuffer*> *bufferList = NULL);

	// This function divides the commBufferList of the buffer manager into two lists: one holding buffers for only
	// intra-segment data transfers and the other for cross-segments data transfers
	void seperateLocalAndRemoteBuffers(int localSegmentTag, 
			List<CommBuffer*> *localBufferList, List<CommBuffer*> *remoteBufferList);

	// this returns IDs of all segments that participate in communications related to the current buffer manager
	virtual std::vector<int> *getParticipantsTags(); 		
};

#endif
