#ifndef _H_comm_stat
#define _H_comm_stat

#include <iostream>
#include <fstream>

#include "../semantics/task_space.h"
#include "../semantics/computation_flow.h"
#include "../semantics/data_access.h"
#include "../static-analysis/sync_stat.h"

#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

/* Whether a synchronization specification results in communication of data or mere synching of PPUs (in 
   current compiler threads) depends on the mapping of LPSes to PPSes and the specific nature of the partition
   functions used to partition the underlying data. Further, once it is determined that a synchronization
   involves communication, information is needed regarding memory allocations of the data in the participating
   LPSes and the confinement within which the processes and/or threads will interact to be in sync. This class
   is presented to hold all information related to communication for a sync-requirement.  
*/
class CommunicationCharacteristics {
  private:
	// the variable that needs to be synchronized
	const char *varName;
	// indicates if communication should be issued for the synchronization
	bool communicationRequired;
	// the participant processes/threads will be inside a same group indicated by their PPU Id for below
	Space *confinementSpace;
	// the sender and receiver LPSes that asked for the synchronization
	Space *senderSyncSpace;
	Space *receiverSyncSpace;
	// the variable may be allocated in ancestor/descendant space other than those that call for the sync
	Space *senderDataAllocatorSpace;
	Space *receiverDataAllocatorSpace;
	// a back pointer to the sync requirement is needed for code generation
	SyncRequirement *syncRequirement;
	
	// When the synchronization dependency is to or from a compiler injected stage the waiting and/or 
	// signaling LPSes for the dependency may be different from the sender/receiver SyncSpace. Therefore
	// the following flag and two other LPS properties are added to the communication characteristics.
	bool waitSignalMayDifferFromSyncLpses;
	Space *signalerSpace;
	Space *waitingSpace;
  public:
	CommunicationCharacteristics(const char *varName);
	const char *getVarName() { return varName; }
	void setCommunicationRequired(bool flag) { communicationRequired = flag; }
	bool isCommunicationRequired() { return communicationRequired; }
	void setConfinementSpace(Space *confinementSpace) { this->confinementSpace = confinementSpace; }
	Space *getConfinementSpace() { return confinementSpace; }
	void setSenderSyncSpace(Space *senderSyncSpace) { this->senderSyncSpace = senderSyncSpace; }
	Space *getSenderSyncSpace() { return senderSyncSpace; }
	void setReceiverSyncSpace(Space *receiverSyncSpace) { this->receiverSyncSpace = receiverSyncSpace; }
	Space *getReceiverSyncSpace() { return receiverSyncSpace; }
	void setSenderDataAllocatorSpace(Space *senderDataAllocatorSpace) {
		this->senderDataAllocatorSpace = senderDataAllocatorSpace;
	}
	Space *getSenderDataAllocatorSpace() { return senderDataAllocatorSpace; }
	void setReceiverDataAllocatorSpace(Space *receiverDataAllocatorSpace) {
		this->receiverDataAllocatorSpace = receiverDataAllocatorSpace;
	}
	Space *getReceiverDataAllocatorSpace() { return receiverDataAllocatorSpace; }
	void setSyncRequirement(SyncRequirement *syncRequirement);
	SyncRequirement *getSyncRequirement();

	void setDifferentWaitSignalerFromSync() { this->waitSignalMayDifferFromSyncLpses = true; }
	bool doesWaitSignalDifferFromSyncLpses() { return waitSignalMayDifferFromSyncLpses; }
	void setSignalerSpace(Space *signaler) { this->signalerSpace = signaler; }
	Space *getSignalerSpace() { return signalerSpace; }
	void setWaitingSpace(Space *waitingSpace) { this->waitingSpace = waitingSpace; } 
	Space *getWaitingSpace() { return waitingSpace; }
	
	// indicates if the underlying communication mechanism implementing the characteristics specified here
	// can be benefited from allocating group resources (for example, the groups of segments interacting 
	// with one-another) as opposed to using a single resource set for all segments
	bool shouldAllocateGroupResources();	
};

#endif
