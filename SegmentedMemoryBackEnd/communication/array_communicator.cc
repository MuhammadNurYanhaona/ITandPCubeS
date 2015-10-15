#include "array_communicator.h"
#include "communicator.h"
#include "comm_buffer.h"
#include "../utils/list.h"
#include <vector>

//------------------------------------------------------- Replication Sync Communicator -------------------------------------------------------/

ReplicationSyncCommunicator::ReplicationSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void ReplicationSyncCommunicator::performTransfer(bool sending) {}

//------------------------------------------------------ Ghost Region Sync Communicator -------------------------------------------------------/

GhostRegionSyncCommunicator::GhostRegionSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void GhostRegionSyncCommunicator::performTransfer() {}

//----------------------------------------------------------- Up Sync Communicator ------------------------------------------------------------/

UpSyncCommunicator::UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void UpSyncCommunicator::sendData() {}

void UpSyncCommunicator::receiveData() {}

//---------------------------------------------------------- Down Sync Communicator -----------------------------------------------------------/

DownSyncCommunicator::DownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void DownSyncCommunicator::sendData() {}

void DownSyncCommunicator::receiveData() {}

//---------------------------------------------------- ----- Cross Sync Communicator ----------------------------------------------------------/

CrossSyncCommunicator::CrossSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void CrossSyncCommunicator::sendData() {}

void CrossSyncCommunicator::receiveData() {}

