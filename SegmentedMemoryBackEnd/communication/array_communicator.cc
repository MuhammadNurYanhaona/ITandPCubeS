#include "array_communicator.h"
#include "confinement_mgmt.h"
#include "communicator.h"
#include "comm_buffer.h"
#include "../utils/list.h"
#include "../utils/utility.h"
#include <vector>

using namespace std;

//------------------------------------------------------- Replication Sync Communicator -------------------------------------------------------/

ReplicationSyncCommunicator::ReplicationSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {

	// Tn a replication sync scenario, the same data is shared among all participating segments. Thus there 
	// should be only one confinement and only one data interchange configurtion in it for the current 
	// segment to do communication for. Consequently, the there should be exactly one communication buffer.
	Assert(bufferList->NumElements() == 1);
}

void ReplicationSyncCommunicator::sendData() {
	
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
        int participants = segmentGroup->getParticipantsCount();
        int myRank = segmentGroup->getRank(localSegmentTag);

        int bcastStatus = 1;
        int bcastStatusList[participants];
        int status = MPI_Allgather(&bcastStatus, 1, MPI_INT, bcastStatusList, 1, MPI_INT, mpiComm);
        if (status != MPI_SUCCESS) {
                cout << "Segment " << localSegmentTag << ": ";
                cout << "could not inform others about being the broadcast source\n";
                exit(EXIT_FAILURE);
        }

	CommBuffer *buffer = commBufferList->Nth(0);
	int bufferSize = buffer->getBufferSize();
	char *data = buffer->getData();

	status = MPI_Bcast(data, bufferSize, MPI_CHAR, myRank, mpiComm);
        if (status != MPI_SUCCESS) {
                cout << "Segment " << localSegmentTag << ": could not broadcast replicated update\n";
                exit(EXIT_FAILURE);
        }
}

void ReplicationSyncCommunicator::receiveData() {
	
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
        int participants = segmentGroup->getParticipantsCount();

        int bcastStatus = 0;
        int bcastStatusList[participants];
        int status = MPI_Allgather(&bcastStatus, 1, MPI_INT, bcastStatusList, 1, MPI_INT, mpiComm);
        if (status != MPI_SUCCESS) {
                cout << "Segment " << localSegmentTag << ": could not find the broadcast source\n";
                exit(EXIT_FAILURE);
        }

        int broadcaster = -1;
        for (int i = 0; i < participants; i++) {
                if (bcastStatusList[i] == 1) {
                        broadcaster = i;
                        break;
                }
        }
        if (broadcaster == -1) {
                cout << "Segment " << localSegmentTag << ": none is making the broadcast\n";
	} else {
		CommBuffer *buffer = commBufferList->Nth(0);
		int bufferSize = buffer->getBufferSize();
		char *data = buffer->getData();

		status = MPI_Bcast(data, bufferSize, MPI_CHAR, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": did not receive broadcast of replicated data\n";
			exit(EXIT_FAILURE);
		}
	}
}


//------------------------------------------------------ Ghost Region Sync Communicator -------------------------------------------------------/

GhostRegionSyncCommunicator::GhostRegionSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {

	// for ghost region sync, each buffer should have one sender segment and one receiver segment (they can be the same)
	// the implementation does not support replication over ghost regions (probably that will never make sense either to
	// have in a program)	
	for (int i = 0; i < bufferList->NumElements(); i++) {
		DataExchange *exchange = bufferList->Nth(i)->getExchange();
		Participant *sender = exchange->getSender();
		if (sender->getSegmentTags().size() != 1) {
			cout << "Segment" << localSegmentTag << ":more than one sender on a ghost region overlap\n";
			exit(EXIT_FAILURE);
		}
		Participant *receiver = exchange->getReceiver();
		if (receiver->getSegmentTags().size() != 1) {
			cout << "Segment" << localSegmentTag << ":more than one receiver on a ghost region overlap\n";
			exit(EXIT_FAILURE);
		}
	}
}

void GhostRegionSyncCommunicator::setupCommunicator() {
	std::vector<int> *participants = getParticipantsTags();
        segmentGroup = new SegmentGroup(*participants);
        delete participants;
}

void GhostRegionSyncCommunicator::performTransfer() {
	
	List<CommBuffer*> *localBufferList = new List<CommBuffer*>;
	List<CommBuffer*> *remoteBufferList = new List<CommBuffer*>;
	seperateLocalAndRemoteBuffers(localSegmentTag, localBufferList, remoteBufferList);
	for (int i = 0; i < localBufferList->NumElements(); i++) {
		CommBuffer *localBuffer = localBufferList->Nth(i);
		// for local buffers just update the operating memory with buffer's content; note that reading data into
		// the buffer is not needed as that has been already done for all buffers 
		localBuffer->writeData();
	}
	delete localBufferList;

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
        int participants = segmentGroup->getParticipantsCount();
        int myRank = segmentGroup->getRank(localSegmentTag);

	// first set up the receiver buffers
	List<CommBuffer*> *remoteReceiveBuffers = getSortedList(true, remoteBufferList);
	int remoteRecvs = remoteReceiveBuffers->NumElements();
	MPI_Request *recvRequests = new MPI_Request[remoteRecvs];
	for (int i = 0; i < remoteRecvs ; i++) {
		CommBuffer *buffer = remoteReceiveBuffers->Nth(i);
		int bufferSize = buffer->getBufferSize();
		char *data = buffer->getData();
		DataExchange *exchange = buffer->getExchange();
		Participant *sender = exchange->getSender();
		int senderSegment = sender->getSegmentTags()[0];
		int senderRank = segmentGroup->getRank(senderSegment);	
		int status = MPI_Irecv(data, bufferSize, MPI_INT, senderRank, 0, mpiComm, &recvRequests[i]);
                if (status != MPI_SUCCESS) {
                	cout << "Segment " << localSegmentTag << ": could not issue asynchronous receive\n";
			exit(EXIT_FAILURE);
		}
	}


	// then do the sends
	List<CommBuffer*> *remoteSendBuffers = getSortedList(true, remoteBufferList);
	int remoteSends = remoteSendBuffers->NumElements();
	MPI_Request *sendRequests = new MPI_Request[remoteSends];
	for (int i = 0; i < remoteSends; i++) {
		CommBuffer *buffer = remoteSendBuffers->Nth(i);
		int bufferSize = buffer->getBufferSize();
		char *data = buffer->getData();
		DataExchange *exchange = buffer->getExchange();
		Participant *receiver = exchange->getReceiver();
		int receiverSegment = receiver->getSegmentTags()[0];
		int receiverRank = segmentGroup->getRank(receiverSegment);	
		int status = MPI_Isend(data, bufferSize, MPI_INT, receiverRank, 0, mpiComm, &recvRequests[i]);
                if (status != MPI_SUCCESS) {
                	cout << "Segment " << localSegmentTag << ": could not issue asynchronous send\n";
			exit(EXIT_FAILURE);
		}
	}

	// wait for all receives to finish and write data back to operating memory from receive buffers
	int status = MPI_Waitall(remoteRecvs, recvRequests, MPI_STATUSES_IGNORE);
	if (status != MPI_SUCCESS) {
		cout << "Segment "<< localSegmentTag << ": some of the asynchronous receives failed\n";
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < remoteRecvs; i++) {
		CommBuffer *buffer = remoteReceiveBuffers->Nth(i);
		buffer->writeData();
	}

	// wait for all sends to finish
	status = MPI_Waitall(remoteSends, sendRequests, MPI_STATUSES_IGNORE);
	if (status != MPI_SUCCESS) {
		cout << "Segment " << localSegmentTag << ": some of the asynchronous sends failed\n";
		exit(EXIT_FAILURE);
	}

	// cleanup data structures before returning
 	delete remoteBufferList;
	delete remoteSendBuffers;
	delete remoteReceiveBuffers;
	delete recvRequests;
	delete sendRequests;
}	

//----------------------------------------------------------- Up Sync Communicator ------------------------------------------------------------/

UpSyncCommunicator::UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {
	
	// there should be just one receiver in an up sync communicator
	Assert(receiverCount == 1);

	commMode = UNKNOWN_COLLECTIVE;
	gatherBuffer = NULL;
	displacements = NULL;
}

UpSyncCommunicator::~UpSyncCommunicator() {
	if (gatherBuffer != NULL) {
		delete gatherBuffer;
		delete displacements;
		delete receiveCounts;
	}
}

void UpSyncCommunicator::setupCommunicator() {
	
	Communicator::setupCommunicator();

	CommBuffer *buffer = commBufferList->Nth(0);
	int receiverSegment = buffer->getExchange()->getReceiver()->getSegmentTags()[0]; 
	if (receiverSegment == localSegmentTag) {
		
		int gather = (commBufferList->NumElements() > 1) ? 1 : 0;

		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int participants = segmentGroup->getParticipantsCount();
		int myRank = segmentGroup->getRank(localSegmentTag);
		
		int status = MPI_Bcast(&gather, 1, MPI_INT, myRank, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": could not broadcast communicator setting\n";
			exit(EXIT_FAILURE);
		}
		
		commMode = (gather == 1) ? GATHER_V : SEND_RECEIVE;
		if (commMode == GATHER_V) allocateAndLinkGatherBuffer();
	} else {
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int participants = segmentGroup->getParticipantsCount();
		int myRank = segmentGroup->getRank(localSegmentTag);
		int broadcaster = segmentGroup->getRank(receiverSegment);

		int gather;
		int status = MPI_Bcast(&gather, 1, MPI_INT, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": did not receive communicator setting broadcast\n";
			exit(EXIT_FAILURE);
		}
		commMode = (gather == 1) ? GATHER_V : SEND_RECEIVE;
	}
}

void UpSyncCommunicator::sendData() {
	
	// note that the logic of confinement and upward sync enforce that there is just one communication buffer in each
	// sender segment
	List<CommBuffer*> *sendBuffers = getSortedList(false);
	CommBuffer *sendBuffer = sendBuffers->Nth(0);
	char *data = sendBuffer->getData();
	int bufferSize = sendBuffer->getBufferSize();
	DataExchange *exchange = sendBuffer->getExchange();

	// for the local data transfer case, just write data from buffer to operating memory as reading has been already 
	// taken care of
	if (exchange->isIntraSegmentExchange(localSegmentTag)) {
		sendBuffer->writeData();
	} else {
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int participants = segmentGroup->getParticipantsCount();
		int receiverSegment = exchange->getReceiver()->getSegmentTags()[0];
		int receiver = segmentGroup->getRank(receiverSegment);

		if (commMode == SEND_RECEIVE) {
			int status = MPI_Send(data, bufferSize, MPI_CHAR, receiver, 0, mpiComm);
                	if (status != MPI_SUCCESS) {
                        	cout << "Segment "  << localSegmentTag << ": could not send update to upper level\n";
                        	exit(EXIT_FAILURE);
                	}
		} else {
			MPI_Comm mpiComm = segmentGroup->getCommunicator();
			int receiverSegment = exchange->getReceiver()->getSegmentTags()[0];
			int receiver = segmentGroup->getRank(receiverSegment);
			int status = MPI_Gatherv(data, bufferSize, MPI_CHAR, 
					NULL, NULL, NULL, MPI_CHAR, receiver, mpiComm);
                	if (status != MPI_SUCCESS) {
                        	cout << "Segment " << localSegmentTag << ": could not participate in data gathering\n";
                        	exit(EXIT_FAILURE);
                	}
		}
	}
}

void UpSyncCommunicator::receiveData() {

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int participants = segmentGroup->getParticipantsCount();
	int myRank = segmentGroup->getRank(localSegmentTag);

	if (commMode == SEND_RECEIVE) {
		// in the send-receive mode there is just one communication buffer in the receiver
		CommBuffer *buffer = commBufferList->Nth(0);
		char *data = buffer->getData();
		int bufferSize = buffer->getBufferSize();
		int status = MPI_Recv(data, bufferSize, MPI_CHAR, MPI_ANY_SOURCE, 0, mpiComm, MPI_STATUS_IGNORE);
                if (status != MPI_SUCCESS) {
                        cout << "Segment " << localSegmentTag << "could not receive up-sync update from unknown source\n";
                        exit(EXIT_FAILURE);
                }
	} else {
		char dummyBuffer;
		int status = MPI_Gatherv(&dummyBuffer, 0, MPI_CHAR,
                                gatherBuffer, receiveCounts, displacements, MPI_CHAR, myRank, mpiComm);
                if (status != MPI_SUCCESS) {
                        cout << "Segment " << localSegmentTag << ": could not gather data\n";
                        exit(EXIT_FAILURE);
                }
	}
}

void UpSyncCommunicator::allocateAndLinkGatherBuffer() {
	
	List<CommBuffer*> *receiveList = getSortedList(true);
	List<CommBuffer*> *localList = new List<CommBuffer*>;
	List<CommBuffer*> *remoteList = new List<CommBuffer*>;
	seperateLocalAndRemoteBuffers(localSegmentTag, localList, remoteList);
	
	// local communication buffers need not be updated
	delete localList;
	
	int gatherBufferSize = 0;
	for (int i = 0; i < remoteList->NumElements(); i++) {
		gatherBufferSize += remoteList->Nth(i)->getBufferSize();
	}
	gatherBuffer = new char[gatherBufferSize];

	int participants = segmentGroup->getParticipantsCount();
	displacements = new int[participants];
	receiveCounts = new int[participants];

	// both displacements and receiveCount vectors must have entries for everyone in the segment group
	for (int i = 0; i < participants; i++) {
		displacements[i] = 0;
		receiveCounts[i] = 0;
	}

	int currentIndex = 0;
	for (int i = 0; i < remoteList->NumElements(); i++) {
		CommBuffer *buffer = remoteList->Nth(i);
		buffer->setData(gatherBuffer + currentIndex);
		
		// note that in the gather mode there should be only one sender segment par communication buffer
		int senderSegment = buffer->getExchange()->getSender()->getSegmentTags()[0];
		int senderRank = segmentGroup->getRank(senderSegment);
		displacements[senderRank] = currentIndex;
		int bufferSize = buffer->getBufferSize();
		receiveCounts[senderRank] = bufferSize;
		currentIndex += bufferSize;
	}

	delete receiveList;
	delete remoteList;
}

//---------------------------------------------------------- Down Sync Communicator -----------------------------------------------------------/

DownSyncCommunicator::DownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void DownSyncCommunicator::setupCommunicator() {}

void DownSyncCommunicator::sendData() {}

void DownSyncCommunicator::receiveData() {}

//---------------------------------------------------- ----- Cross Sync Communicator ----------------------------------------------------------/

CrossSyncCommunicator::CrossSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int senderCount, 
		int receiverCount, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, dependencyName, senderCount, receiverCount) {}

void CrossSyncCommunicator::setupCommunicator() {
	std::vector<int> *participants = getParticipantsTags();
        segmentGroup = new SegmentGroup(*participants);
        delete participants;
}

void CrossSyncCommunicator::sendData() {}

void CrossSyncCommunicator::receiveData() {}

