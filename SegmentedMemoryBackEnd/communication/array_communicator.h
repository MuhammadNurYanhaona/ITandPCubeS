#ifndef _H_array_communicator
#define _H_array_communicator

/* This library holds implementation of data communication for different kinds of synchronization scenarios involving arrays
 */

#include "comm_buffer.h"
#include "communicator.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"

// some communicators doing collective communications need to setup the mode of communication they are going to use before 
// any communication can be done; that is why this enum is required 
enum CommMode {	BROADCAST, 
		GATHER_V, 
		SCATTER_V, 
		SEND_RECEIVE, // to receive from one of a list of senders
		UNKNOWN_COLLECTIVE
};

// communicator class for the scenario of synchronization a replicated data among the LPUs for a single LPS 
class ReplicationSyncCommunicator : public Communicator {
  public:
	ReplicationSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus, int localReceiverPpus, List<CommBuffer*> *bufferList);

	void sendData();
        void receiveData();
	
	// sender should not wait on receive; this override ensures that
	void afterSend() { iterationNo++; }
};

// communicator class for the scenario of synchronizing overlapping boundary regions among LPUs of a single LPS
class GhostRegionSyncCommunicator : public Communicator {
  public:
	GhostRegionSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus, int localReceiverPpus, List<CommBuffer*> *bufferList);

	// ghost region sync does not need a new MPI communicator; this this override is given to just register the segments
	// as participants and use the default MPI communicator
	void setupCommunicator(bool includeNonInteractingSegments);

	void sendData() { performTransfer(); }
        void receiveData() {}
	
	// any segment that sends ghost-region update to someone else receives updates back; so we can combine send-receive
	// within a single function and let the later receive call to be non-halting 
	void afterSend() { iterationNo++; }
	void performTransfer();
};

// communictor class for the scenario of propaging update to a data from LPUs of a lower level LPS to the LPU of a higher 
// level LPS that embody the former LPUs
class UpSyncCommunicator : public Communicator {
  protected:
	// after the communicator setup the mode should be either gather_v or send_receive
	CommMode commMode;
	// when gather_v is used as then we need a collective buffer in the receiver that will hold data received from all 
	// remote senders
	char *gatherBuffer;
	// a displacement vector is also needed to specify which part of the gather buffer should receive data from which 
	// segment
	int *displacements;
	// an array to specify how many elements should be received from each sender segment in the gather mode
	int *receiveCounts;
  public:
	UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, int localReceiverPpus, List<CommBuffer*> *bufferList);
	~UpSyncCommunicator();
	
	// communicator setup needs to be extended to determine the mode of communication
	void setupCommunicator(bool includeNonInteractingSegments);

	void sendData();
        void receiveData();
	
	// the sender segment should not wait on its own update 
	void afterSend() { if (commMode == SEND_RECEIVE) iterationNo++; }
  private:
	// this allocate a gather buffer and link portion of its to different communication buffers so that there is no need
	// for copying data from the gather buffer to a set of communication buffers before the data can be copied from their
	// to operating memory
	void allocateAndLinkGatherBuffer();
};

// communicator class for the scenario opposite to the UpSyncCommunicator
class DownSyncCommunicator : public Communicator {
  protected:
	// after the communicator setup the mode should be either broadcast or scater_v
	CommMode commMode;

	// a flag denoting if the sender side of the communicator is replicated; if YES then one of many will do the data 
	// send each time the communicator is used and there will be an extra step to determine who is the current sender
	bool replicated;
	
	// three variables to be used for scatter_v communication  
	char *scatterBuffer;
	int *sendCounts;
	int *displacements;
  public:
	DownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, int localReceiverPpus, List<CommBuffer*> *bufferList);
	~DownSyncCommunicator();
	
	// communicator setup needs to be extended to determine the mode of communication
	void setupCommunicator(bool includeNonInteractingSegments);

	void sendData();
        void receiveData();
	
	// the sender segment should not wait on its own update 
	void afterSend() { iterationNo++; }
  private:
	// just like in the case of gather-buffer setup in the up-sync-communicator, a scatter-buffer setup is needed for
	// this communicator when scatter_v is used as the form of collective communication 
	void allocateAndLinkScatterBuffer();

	// this function is needed, in particular, when the sender LPS of the underlying dependency has replication in it;
	// then the sender with lowest segment tag setup the communication mode 
	int getFirstSenderInCommunicator();

	// In the replicated mode; all co-operating segments first should identify the sender segment for the current use
	// of the communicator. They invoke this function for this purpose. 
	int discoverSender(bool sendingData);

	// this function is used by the sender segment in the scatter_v communication mode to update its local receive buffer
	// directly without going through the MPI layer
	void updateLocalBufferPart();	
};

// communicator class for the scenario where LPUs of two different LPSes that are not hierarchically related needs to be
// synchronized after an update done on one LPS	 
class CrossSyncCommunicator : public Communicator {
  public:
	CrossSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, int localReceiverPpus, List<CommBuffer*> *bufferList);

	// like ghost region sync, cross-sync does not need a new MPI communicator; so this override uses the default MPI
	// communicator
	void setupCommunicator(bool includeNonInteractingSegments);

	void sendData();
        void receiveData();

	// Because the way MPI works, the sends can get deadlocked if there is/are receives on the receiving segments. But
	// it may happen that all segments are trying to send data to others. Consequently there will be no receive issued
	// by any of them. To overcome this problem, during send, a segment checks if it is supposed to receive any data 
	// subsequently from other segments and issue aynchronous receives before issuing its own sends (that will be asyn-
	// chronous too). This method issues the receives but does not wait for them to finish. Rather, it returns the 
	// array of MPI requests status that the invoker can wait on when needed.
	MPI_Request *issueAsyncReceives(List<CommBuffer*> *remoteReceiveBuffers);
	
	// due to the asynchronous receive setup; if send is invoked no subsequent receive is needed for the same iteration
	void afterSend() { iterationNo++; }
};

#endif
