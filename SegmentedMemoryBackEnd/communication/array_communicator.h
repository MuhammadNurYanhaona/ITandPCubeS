#ifndef _H_array_communicator
#define _H_array_communicator

/* This library holds implementation of data communication for different kinds of synchronization scenarios involving arrays
 */

#include "comm_buffer.h"
#include "communicator.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"

// communicator class for the scenario of synchronization a replicated data among the LPUs for a single LPS 
class ReplicationSyncCommunicator : public Communicator {
  public:
	ReplicationSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

	void sendData() { performTransfer(true); }
        void receiveData() { performTransfer(false); }
	
	// sender should not wait on receive; this override ensures that
	void afterSend() { iterationNo++; }
	
	// the nature of data transfer for send and receive are the same for replication; only the particular role a segment
	// plays differs  
	void performTransfer(bool sending);
};

// communicator class for the scenario of synchronizing overlapping boundary regions among LPUs of a single LPS
class GhostRegionSyncCommunicator : public Communicator {
  public:
	GhostRegionSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

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
  public:
	UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

	void sendData();
        void receiveData();
	
	// the sender segment should not wait on its own update 
	void afterSend() { iterationNo++; }
};

// communicator class for the scenario opposite to the UpSyncCommunicator
class DownSyncCommunicator : public Communicator {
  public:
	DownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, int receiverCount, List<CommBuffer*> *bufferList);

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

	void sendData();
        void receiveData();
};

#endif
