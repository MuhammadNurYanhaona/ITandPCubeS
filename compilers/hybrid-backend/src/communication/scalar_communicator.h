/* Scalar variables are not partitioned like arrays so the logic of confinements, data exchanges, and comm-buffers that
 * we developed for arrays does not adjust well for scalar variable synchronizations. The bulk-synchronous communicator
 * is, however, applicable to scalar variables. So this header file provides extensions to the communicator from commu-
 * nicator.h for different forms of scalar variables synchronizations.
 */

#ifndef _H_scalar_communicator
#define _H_scalar_communicator

#include "communicator.h"
#include <vector>

class ScalarCommunicator : public Communicator {
  protected:
	// property holding memory reference of the scalar variable 
	char *dataBuffer;
	// size of the scalar variable in terms of number of characters
	int dataSize;
	// the communicator is not active when there is none except the current segment participating in synchronization
	bool active;
	// tags of sender and receiver segments
	std::vector<int> senderSegmentTags;
	std::vector<int> receiverSegmentTags;
	// temporary variables needed to determine the tags of sender and receiver segments
	bool hasLocalSender;
	bool hasLocalReceiver;
  public:
	ScalarCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus,
		int localReceiverPpus, 
		int dataSize);
	virtual ~ScalarCommunicator() {}

	// before a data send/receive the memory address of the scalar variable should be copied into the communicator
	// using this funtion
	void setDataBufferReference(void *dataBuffer) { this->dataBuffer = reinterpret_cast<char*>(dataBuffer); }

	// MPI communicator setup process should be updated for scaler IT communicators as they do not involve processing
	// of communication buffers in search of participating segments  
	void setupCommunicator(bool includeNonInteractingSegments);

	// there is no buffer preparation or post-processing for scalar variables
	void prepareBuffersForSend() {}
        void processBuffersAfterReceive() {}
		
	// since there is just one instance of each scalar variable; local data interchange is inapplicable for them; the
	// send and receive data functions should only be used when there are other participating segments (when the 
	// active flag is true); so override has been provided to do actual transfers only in multi-party situations. 
	void sendData() { if (active) send(); }
	void receiveData() { if (active) receive(); }

	// for scalar variables only one of send and receive is enough for synchronization; therefore, the after send
	// functions increases the iteration number of the communicator to let any subsequent receive request from PPUs
	// on the same iteration will bypass the communicator
	void afterSend() { iterationNo++; }
	
	// functions subclasses should override to implement specific forms of scalar synchronization
	virtual void send() = 0;
	virtual void receive() = 0;
};

// Note that scalar variables are replicated throughout the partition hierarchy. So every update is supposed to be syn-
// chronized throught the machine in all segments. The flow of execution of a task, however, restricts where should an
// update should propagate at a specific instance. Therefore, we have three scenarios of restricted updates for scalar
// variables as handled by the following three subclasses. 

// handles a scenario where an update done within a segment needs to be shared with all other segments executing compute
// stages belonging to the same LPS where the update has taken place
class ScalarReplicaSyncCommunicator : public ScalarCommunicator {
  public:
	ScalarReplicaSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus,
		int localReceiverPpus, 
		int dataSize);
	void send();
	void receive();
};

// handles a scenario where update done by a segment executing code for a lower level LPS needs to be propagated to the
// segment executing codes for a higher level LPS that involving the same shared scalar variable
class ScalarUpSyncCommunicator : public ScalarCommunicator {
  public:
	ScalarUpSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus,
		int localReceiverPpus, 
		int dataSize);
	void send();
	void receive();
};

// handles a scenario that is exactly opposite to the UpSync scenario; here the updates propage from a segment executing 
// compute stages of a higher level LPS to segments executing stages of a lower level LPS; the effect of this synchroni-
// zation is similar to that of the replication sync but the difference is the the source of update is fixed here
class ScalarDownSyncCommunicator : public ScalarCommunicator {
  public:
	ScalarDownSyncCommunicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus,
		int localReceiverPpus, 
		int dataSize);
	void send();
	void receive();
};

#endif
