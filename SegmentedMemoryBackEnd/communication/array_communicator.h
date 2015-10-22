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
		int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

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
		int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

	// ghost region sync does not need a new MPI communicator; this this override is given to just register the segments
	// as participants and use the default MPI communicator
	void setupCommunicator();

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
	int gatherBufferSize;
	// a displacement vector is also needed to specify which part of the gather buffer should receive data from which 
	// segment
	int *displacements;
	// an array to specify how many elements should be received from each sender segment in the gather mode
	int *receiveCounts;
  public:
	UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, int receiverCount, List<CommBuffer*> *bufferList);
	~UpSyncCommunicator();
	
	// communicator setup needs to be extended to determine the mode of communication
	void setupCommunicator();

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
  public:
	DownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, int receiverCount, List<CommBuffer*> *bufferList);
	
	// communicator setup needs to be extended to determine the mode of communication
	void setupCommunicator();

	void sendData();
        void receiveData();
	
	// the sender segment should not wait on its own update 
	void afterSend() { iterationNo++; }
};

// communicator class for the scenario where LPUs of two different LPSes that are not hierarchically related needs to be
// synchronized after an update done on one LPS	 
class CrossSyncCommunicator : public Communicator {
  public:
	CrossSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

	// like ghost region sync, cross-sync does not need a new MPI communicator
	void setupCommunicator();

	void sendData();
        void receiveData();
};

#endif
