#include "array_communicator.h"
#include "confinement_mgmt.h"
#include "communicator.h"
#include "comm_buffer.h"
#include "../utils/list.h"
#include "../utils/utility.h"

#include <vector>
#include <climits>
#include <mpi.h>

using namespace std;

//------------------------------------------------------- Replication Sync Communicator -------------------------------------------------------/

ReplicationSyncCommunicator::ReplicationSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, 
		int localReceiverPpus, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, 
			dependencyName, localSenderPpus, localReceiverPpus) {

	// Tn a replication sync scenario, the same data is shared among all participating segments. Thus there 
	// should be only one confinement and only one data interchange configurtion in it for the current 
	// segment to do communication for. Consequently, there should be exactly one communication buffer.
	Assert(bufferList->NumElements() == 1);

	this->commBufferList = bufferList;
}

void ReplicationSyncCommunicator::sendData() {

	//*logFile << "\tReplication-sync communicator is sending data for " << dependencyName << "\n";
	//logFile->flush();
	
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
	
	//*logFile << "\tReplication-sync communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}

void ReplicationSyncCommunicator::receiveData() {
	
	//*logFile << "\tReplication-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();

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
	
	//*logFile << "\tReplication-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
}


//------------------------------------------------------ Ghost Region Sync Communicator -------------------------------------------------------/

GhostRegionSyncCommunicator::GhostRegionSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, 
		int localReceiverPpus, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, 
			dependencyName, localSenderPpus, localReceiverPpus) {

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

	this->commBufferList = bufferList;
}

void GhostRegionSyncCommunicator::setupCommunicator(bool includeNonInteractingSegments) {
	std::vector<int> *participants = getParticipantsTags();
        segmentGroup = new SegmentGroup(*participants);
        delete participants;
	*logFile << "\tNo MPI resource setup was needed for Ghost-region Sync Communicator for ";
	*logFile << dependencyName << "\n";
	logFile->flush();
}

void GhostRegionSyncCommunicator::performTransfer() {
	
	//*logFile << "\tGhost-sync communicator is communicating data for " << dependencyName << "\n";
	//logFile->flush();
	
	List<CommBuffer*> *localBufferList = new List<CommBuffer*>;
	List<CommBuffer*> *remoteBufferList = new List<CommBuffer*>;
	seperateLocalAndRemoteBuffers(localSegmentTag, localBufferList, remoteBufferList);

	for (int i = 0; i < localBufferList->NumElements(); i++) {
		CommBuffer *localBuffer = localBufferList->Nth(i);
		// for local buffers just update the operating memory with buffer's content; note that reading data into
		// the buffer is not needed as that has been already done for all buffers 
		localBuffer->writeData(false, *logFile);
	}
	delete localBufferList;

	if (remoteBufferList->NumElements() == 0) {
                delete remoteBufferList;
                //*logFile << "\tGhost-sync communicator locally exchanged data for " << dependencyName << "\n";
                //logFile->flush();
                return;
        }

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
		vector<int> senderTags = sender->getSegmentTags();
		Assert(senderTags.size() == 1);
		int senderSegment = senderTags[0];
		Assert(senderSegment != localSegmentTag);
		int senderRank = segmentGroup->getRank(senderSegment);	
		int status = MPI_Irecv(data, bufferSize, MPI_CHAR, senderRank, 0, mpiComm, &recvRequests[i]);
                if (status != MPI_SUCCESS) {
                	cout << "Segment " << localSegmentTag << ": could not issue asynchronous receive\n";
			exit(EXIT_FAILURE);
		}
	}


	// then do the sends
	List<CommBuffer*> *remoteSendBuffers = getSortedList(false, remoteBufferList);
	int remoteSends = remoteSendBuffers->NumElements();
	MPI_Request *sendRequests = new MPI_Request[remoteSends];
	for (int i = 0; i < remoteSends; i++) {
		CommBuffer *buffer = remoteSendBuffers->Nth(i);
		int bufferSize = buffer->getBufferSize();
		char *data = buffer->getData();
		DataExchange *exchange = buffer->getExchange();
		Participant *receiver = exchange->getReceiver();
		vector<int> receiverTags = receiver->getSegmentTags();
		Assert(receiverTags.size() == 1);
		int receiverSegment = receiverTags[0];
		Assert(receiverSegment != localSegmentTag);
		int receiverRank = segmentGroup->getRank(receiverSegment);	
		int status = MPI_Isend(data, bufferSize, MPI_CHAR, receiverRank, 0, mpiComm, &sendRequests[i]);
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
		buffer->writeData(false, *logFile);
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
	delete[] recvRequests;
	delete[] sendRequests;
	
	//*logFile << "\tGhost-sync communicator sent-received data for " << dependencyName << "\n";
	//logFile->flush();
}	

//----------------------------------------------------------- Up Sync Communicator ------------------------------------------------------------/

UpSyncCommunicator::UpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, 
		int localReceiverPpus, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, 
			dependencyName, localSenderPpus, localReceiverPpus) {
	
	// there should be just one receiver in an up sync communicator
	for (int i = 0; i < bufferList->NumElements(); i++) {
		CommBuffer *buffer = bufferList->Nth(i);
		Participant *receiver = buffer->getExchange()->getReceiver();
		if (receiver->getSegmentTags().size() != 1) {
			cout << "Segment " << localSegmentTag;
			cout << ": there cannot be more than one receiver segment on up-sync\n";
			exit(EXIT_FAILURE);
		}
	}

	commMode = UNKNOWN_COLLECTIVE;
	gatherBuffer = NULL;
	displacements = NULL;
	receiveCounts = NULL;

	this->commBufferList = bufferList;
}

UpSyncCommunicator::~UpSyncCommunicator() {
	if (gatherBuffer != NULL) {
		delete[] gatherBuffer;
		delete[] displacements;
		delete[] receiveCounts;
	}
}

void UpSyncCommunicator::setupCommunicator(bool includeNonInteractingSegments) {
	
	Communicator::setupCommunicator(includeNonInteractingSegments);

	*logFile << "\tsetting up the mode for up-sync communicator for " << dependencyName << "\n";
	logFile->flush();

	struct timeval start;
        gettimeofday(&start, NULL);

	CommBuffer *buffer = commBufferList->Nth(0);
	int receiverSegment = buffer->getExchange()->getReceiver()->getSegmentTags()[0]; 
	if (receiverSegment == localSegmentTag) {
		
		int gather = (commBufferList->NumElements() > 1) ? 1 : 0;

		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int myRank = segmentGroup->getRank(localSegmentTag);
		
		int status = MPI_Bcast(&gather, 1, MPI_INT, myRank, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": could not broadcast up-sync communicator setting\n";
			exit(EXIT_FAILURE);
		}
		
		commMode = (gather == 1) ? GATHER_V : SEND_RECEIVE;
		if (commMode == GATHER_V) allocateAndLinkGatherBuffer();
	} else {
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int broadcaster = segmentGroup->getRank(receiverSegment);

		int gather;
		int status = MPI_Bcast(&gather, 1, MPI_INT, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": did not receive communicator setting broadcast\n";
			exit(EXIT_FAILURE);
		}
		commMode = (gather == 1) ? GATHER_V : SEND_RECEIVE;
	}

	struct timeval end;
        gettimeofday(&end, NULL);
        commStat->addCommResourcesSetupTime(dependencyName, start, end);
	
	*logFile << "\tmode setup done for up-sync communicator for " << dependencyName << "\n";
	logFile->flush();
}

void UpSyncCommunicator::sendData() {
	
	//*logFile << "\tUp-sync communicator is sending data for " << dependencyName << "\n";
	//logFile->flush();
	
	// note that the logic of confinement and upward sync enforce that there is just one communication buffer in each
	// sender segment
	List<CommBuffer*> *sendBuffers = getSortedList(false);
	CommBuffer *sendBuffer = sendBuffers->Nth(0);
	DataExchange *exchange = sendBuffer->getExchange();

	// for the local data transfer case, just write data from buffer to operating memory as reading has been already 
	// taken care of
	if (exchange->getReceiver()->hasSegmentTag(localSegmentTag)) {
		sendBuffer->writeData(false, *logFile);
	} else {
		char *data = sendBuffer->getData();
		int bufferSize = sendBuffer->getBufferSize();
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
			int status = MPI_Gatherv(data, bufferSize, MPI_CHAR, 
					NULL, NULL, NULL, MPI_CHAR, receiver, mpiComm);
                	if (status != MPI_SUCCESS) {
                        	cout << "Segment " << localSegmentTag << ": could not participate in data gathering\n";
                        	exit(EXIT_FAILURE);
                	}
		}
	}
	
	//*logFile << "\tUp-sync communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}

void UpSyncCommunicator::receiveData() {

	//*logFile << "\tUp-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
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
	
	//*logFile << "\tUp-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
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

	// both displacements and receiveCounts vectors must have entries for everyone in the segment group
	for (int i = 0; i < participants; i++) {
		displacements[i] = 0;
		receiveCounts[i] = 0;
	}

	int currentIndex = 0;
	for (int i = 0; i < remoteList->NumElements(); i++) {
		CommBuffer *buffer = remoteList->Nth(i);
		buffer->setData(gatherBuffer + currentIndex);
		
		// note that in the gather mode there should be only one sender segment per communication buffer
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
                int localSenderPpus, 
		int localReceiverPpus, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, 
			dependencyName, localSenderPpus, localReceiverPpus) {

	this->replicated = false;	
	for (int i = 0; i < bufferList->NumElements(); i++) {
		CommBuffer *buffer = bufferList->Nth(i);
		Participant *sender = buffer->getExchange()->getSender();
		if (sender->getSegmentTags().size() > 1) {
			this->replicated = true;
			break;
		}
	}
	
	commMode = UNKNOWN_COLLECTIVE;
	scatterBuffer = NULL;
	sendCounts = NULL;
	displacements = NULL;

	this->commBufferList = bufferList;
}

DownSyncCommunicator::~DownSyncCommunicator() {
	if (scatterBuffer != NULL) {
		delete[] scatterBuffer;
		delete[] sendCounts;
		delete[] displacements;
	}
}

void DownSyncCommunicator::setupCommunicator(bool includeNonInteractingSegments) {
	
	Communicator::setupCommunicator(includeNonInteractingSegments);
	
	*logFile << "\tsetting up the mode for down-sync communicator for " << dependencyName << "\n";
	logFile->flush();
	
	struct timeval start;
        gettimeofday(&start, NULL);

	int senderTag = getFirstSenderInCommunicator();
	if (senderTag == localSegmentTag) {
		int scatter = (commBufferList->NumElements() > 1) ? 1 : 0;
		
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int myRank = segmentGroup->getRank(localSegmentTag);
		
		int status = MPI_Bcast(&scatter, 1, MPI_INT, myRank, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": could not broadcast down-sync communicator setting\n";
			exit(EXIT_FAILURE);
		}
		
		commMode = (scatter == 1) ? SCATTER_V : BROADCAST;
		if (commMode == SCATTER_V) allocateAndLinkScatterBuffer();
	} else {
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int broadcaster = segmentGroup->getRank(senderTag);

		int scatter;
		int status = MPI_Bcast(&scatter, 1, MPI_INT, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": did not receive communicator setting broadcast\n";
			exit(EXIT_FAILURE);
		}
		commMode = (scatter == 1) ? SCATTER_V : BROADCAST;
	}
	
	struct timeval end;
        gettimeofday(&end, NULL);
        commStat->addCommResourcesSetupTime(dependencyName, start, end);
	
	*logFile << "\tmode setup done for down-sync communicator for " << dependencyName << "\n";
	logFile->flush();
}

void DownSyncCommunicator::sendData() {

	//*logFile << "\tDown-sync communicator is sending data for " << dependencyName << "\n";
	//logFile->flush();
	
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int myRank = segmentGroup->getRank(localSegmentTag);

	// in the replicated mode, set up the broadcaster ID in the receivers before the actual data communication can take place
	if (replicated) discoverSender(true);

	if (commMode == BROADCAST) {

		// in the broadcast transfer mode there is just one buffer content to communicate
		CommBuffer *buffer = commBufferList->Nth(0);

		// issue the MPI broadcast only when there are some other segments waiting to receive the data
		DataExchange *exchange = buffer->getExchange();
		if (!exchange->isIntraSegmentExchange(localSegmentTag)) {
			char *data = buffer->getData();
			int bufferSize = buffer->getBufferSize();
			int status = MPI_Bcast(data, bufferSize, MPI_CHAR, myRank, mpiComm);
			if (status != MPI_SUCCESS) {
				cout << "Segment " << localSegmentTag;
				cout << ": could not broadcast update to lower level LPS\n";
				exit(EXIT_FAILURE);
			}
		}

		// need to update own operating memory data as the sender will bypass the receive call on the communicator
		buffer->writeData(false, *logFile);
	} else {		
		// need to update its own receiving buffer in the scatter mode separately; interchange for others are included
		// in the scatter buffer configuration 
		updateLocalBufferPart();

		char dummyReceive = 0;
		int status = MPI_Scatterv(scatterBuffer, sendCounts, displacements, MPI_CHAR,
                                &dummyReceive, 0, MPI_CHAR, myRank, mpiComm);
                if (status != MPI_SUCCESS) {
                        cout << "Segment" << localSegmentTag << ": could not scatter data update on down-sync\n";
                        exit(EXIT_FAILURE);
                }
	}
	
	//*logFile << "\tDown-sync communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}

void DownSyncCommunicator::receiveData() {
	
	//*logFile << "\tDown-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();

	// there will be one communication buffer per receiver regardless of the communication mode on a down-sync as there 
	// is just one sender
	CommBuffer *buffer = commBufferList->Nth(0);
	char *data = buffer->getData();
	int bufferSize = buffer->getBufferSize();
	
	int senderTag = buffer->getExchange()->getSender()->getSegmentTags()[0];
	int sender = segmentGroup->getRank(senderTag);
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	
	// in the replicated mode, retrieve the broadcaster ID before the actual data communication can take place
	if (replicated) {
		sender = discoverSender(false);
	}
	
	if (commMode == BROADCAST) {
		int status = MPI_Bcast(data, bufferSize, MPI_CHAR, sender, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": did not receive broadcast update on down-sync\n";
			exit(EXIT_FAILURE);
		}
	} else {
		int status = MPI_Scatterv(NULL, NULL, NULL, MPI_CHAR, data, bufferSize, MPI_CHAR, sender, mpiComm);
                if (status != MPI_SUCCESS) {
                        cout << "Segment" << localSegmentTag << ": could not participate in data scattering\n";
                        exit(EXIT_FAILURE);
                }
	}

	//*logFile << "\tDown-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
} 

void DownSyncCommunicator::allocateAndLinkScatterBuffer() {

	List<CommBuffer*> *sendList = getSortedList(false);
	List<CommBuffer*> *localList = new List<CommBuffer*>;
	List<CommBuffer*> *remoteList = new List<CommBuffer*>;
	seperateLocalAndRemoteBuffers(localSegmentTag, localList, remoteList);
	
	// local communication buffers need not be updated to be connected with the accumulator scatter send buffer
	delete localList;

	int scatterBufferSize = 0;
	for (int i = 0; i < remoteList->NumElements(); i++) {
		scatterBufferSize += remoteList->Nth(i)->getBufferSize();
	}
	scatterBuffer = new char[scatterBufferSize];

	int participants = segmentGroup->getParticipantsCount();
	displacements = new int[participants];
	sendCounts = new int[participants];

	// both displacements and receiveCount vectors must have entries for everyone in the segment group
	for (int i = 0; i < participants; i++) {
		displacements[i] = 0;
		sendCounts[i] = 0;
	}
	
	int currentIndex = 0;
	for (int i = 0; i < remoteList->NumElements(); i++) {
		CommBuffer *buffer = remoteList->Nth(i);
		buffer->setData(scatterBuffer + currentIndex);
		int bufferSize = buffer->getBufferSize();
		
		// in the replicated mode a single buffer may be shared by multiple receiver segments; such segments
		// should have the same displacement index
		vector<int> receiverTags = buffer->getExchange()->getReceiver()->getSegmentTags();
		for (int j = 0; j < receiverTags.size(); j++) { 
			int receiver = receiverTags[j];
			if (receiver == localSegmentTag) continue;
			int receiverRank = segmentGroup->getRank(receiver);
			displacements[receiverRank] = currentIndex;
			sendCounts[receiverRank] = bufferSize;
		}

		currentIndex += bufferSize;
	}

	delete sendList;
	delete remoteList;
}

int DownSyncCommunicator::getFirstSenderInCommunicator() {
	int lowestTag = INT_MAX;
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		CommBuffer *buffer = commBufferList->Nth(i);
		Participant *sender = buffer->getExchange()->getSender();
		int firstTag = sender->getSegmentTags()[0];
		if (firstTag < lowestTag) {
			lowestTag = firstTag;
		}	
	}
	return lowestTag;
}

int DownSyncCommunicator::discoverSender(bool sendingData) {

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
        int participants = segmentGroup->getParticipantsCount();
        int myRank = segmentGroup->getRank(localSegmentTag);

	if (sendingData) {
		int bcastStatus = 1;
		int bcastStatusList[participants];
		int status = MPI_Allgather(&bcastStatus, 1, MPI_INT, bcastStatusList, 1, MPI_INT, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": ";
			cout << "could not inform others about being the broadcast source\n";
			exit(EXIT_FAILURE);
		}
		return myRank;
	} else {
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
		return broadcaster;
	}
}

void DownSyncCommunicator::updateLocalBufferPart() {
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		CommBuffer *buffer = commBufferList->Nth(i);
		Participant *receiver = buffer->getExchange()->getReceiver();
		if (receiver->hasSegmentTag(localSegmentTag)) {
			buffer->writeData(false, *logFile);
			break;
		}
	}
}

//---------------------------------------------------- ----- Cross Sync Communicator ----------------------------------------------------------/

CrossSyncCommunicator::CrossSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                int localSenderPpus, 
		int localReceiverPpus, List<CommBuffer*> *bufferList) 
		: Communicator(localSegmentTag, 
			dependencyName, localSenderPpus, localReceiverPpus) {

	this->commBufferList = bufferList;
}

void CrossSyncCommunicator::setupCommunicator(bool includeNonInteractingSegments) {
	std::vector<int> *participants = getParticipantsTags();
        segmentGroup = new SegmentGroup(*participants);
        delete participants;
	*logFile << "\tNo MPI resource setup was needed for Cross-Sync Communicator for " << dependencyName << "\n";
	logFile->flush();
}
 
void CrossSyncCommunicator::sendData() {

	//*logFile << "\tCross-sync communicator is sending (and receiving) data for " << dependencyName << "\n";
	//logFile->flush();
	
	List<CommBuffer*> *localBuffers = new List<CommBuffer*>;
	List<CommBuffer*> *remoteBuffers = new List<CommBuffer*>;

	seperateLocalAndRemoteBuffers(localSegmentTag, localBuffers, remoteBuffers);
	
	// write the local buffers into operating memory; as in other cases, reading has been done already in some earlier function call
	for (int i = 0; i < localBuffers->NumElements(); i++) {
		localBuffers->Nth(i)->writeData(false, *logFile);
	}
	delete localBuffers;

	// issue asynchronous receives first, when applicable
	MPI_Request *receiveRequests = NULL;
	// Note that in some receiver buffers, the current segment may be listed as sender among the group of possible senders. This 
	// happens when the sender side of the cross-sync has replication somewhere in the partition hierarchy. Therefore, the current
	// segment may be sending data to itself or receiving updates from some other segment for a replicated data part. If the send
	// is invoked by the current segment, as it has been done here, then the assumption is that the current segment has updated the
	// replicated data. Hence, communication buffers having the current segment as a possible sender should be removed from the
	// list before issuing remote receives. 
	List<CommBuffer*> *activeReceives = getSortedList(true, remoteBuffers);
	List<CommBuffer*> *remoteReceives = new List<CommBuffer*>;
	for (int i = 0; i < activeReceives->NumElements(); i++) {
		CommBuffer *buffer = activeReceives->Nth(i);
		if (!buffer->isSendActivated()) {
			remoteReceives->Append(buffer);
		}
	}
	int receiveCount = remoteReceives->NumElements();
	if (receiveCount > 0) {
		receiveRequests = issueAsyncReceives(remoteReceives);		
	}
	
	// calculate the number of MPI requests that should be issued
	List<CommBuffer*> *remoteSends = getSortedList(false, remoteBuffers);
	int sendCount = 0;
	for (int i = 0; i < remoteSends->NumElements(); i++) {
		CommBuffer *buffer = remoteSends->Nth(i);
		Participant *participant = buffer->getExchange()->getReceiver();
		sendCount += participant->getSegmentTags().size();
		if (participant->hasSegmentTag(localSegmentTag)) {
			sendCount--;
		}		
	}

	// issue the sends
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	MPI_Request *sendRequests = new MPI_Request[sendCount];
	int requestIndex = 0;
	for (int i = 0; i < remoteSends->NumElements(); i++) {
		CommBuffer *buffer = remoteSends->Nth(i);
		int bufferTag = buffer->getBufferTag();
		vector<int> receiverSegments = buffer->getExchange()->getReceiver()->getSegmentTags();
		for (int j = 0; j < receiverSegments.size(); j++) {
			int segmentTag = receiverSegments.at(j);
			// if data to be sent to a remote segment is also replicated locally then write the buffer in the local
			// operating memory
			if (segmentTag == localSegmentTag) {
				buffer->writeData(false, *logFile);
				continue;
			}

			int receiver =segmentGroup->getRank(segmentTag);
			char *data = buffer->getData();
			int bufferSize = buffer->getBufferSize();
			int status = MPI_Isend(data, bufferSize, MPI_CHAR, 
					receiver, bufferTag, mpiComm, &sendRequests[requestIndex]);
			if (status != MPI_SUCCESS) {
				cout << "Segment " << localSegmentTag << ": could not issue asynchronous send\n";
				exit(EXIT_FAILURE);
			}
			requestIndex++;
		}
	}
	
	// wait for all receives to finish and write data back to operating memory from receive buffers
	if (receiveCount > 0) {
		int status = MPI_Waitall(receiveCount, receiveRequests, MPI_STATUSES_IGNORE);
		if (status != MPI_SUCCESS) {
			cout << "Segment "<< localSegmentTag << ": some of the asynchronous receives failed\n";
			exit(EXIT_FAILURE);
		}
		for (int i = 0; i < remoteReceives->NumElements(); i++) {
			CommBuffer *buffer = remoteReceives->Nth(i);
			buffer->writeData(false, *logFile);
		}
	}

	// wait for all sends to finish
	int status = MPI_Waitall(sendCount, sendRequests, MPI_STATUSES_IGNORE);
	if (status != MPI_SUCCESS) {
		cout << "Segment " << localSegmentTag << ": some of the asynchronous sends failed\n";
		exit(EXIT_FAILURE);
	}

	delete remoteBuffers;
	delete remoteSends;
	delete activeReceives;
	delete remoteReceives;
	if (receiveCount > 0) delete[] receiveRequests;
	delete[] sendRequests;
	
	//*logFile << "\tCross-sync communicator sent (and received) data for " << dependencyName << "\n";
	//logFile->flush();
}

void CrossSyncCommunicator::receiveData() {

	//*logFile << "\tCross-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();
	
	List<CommBuffer*> *localBuffers = new List<CommBuffer*>;
	List<CommBuffer*> *remoteBuffers = new List<CommBuffer*>;
	seperateLocalAndRemoteBuffers(localSegmentTag,localBuffers, remoteBuffers);
	
	// local buffers has been taken care of in the sendData() function
	delete localBuffers;
	
	// issue asynchronous receives
	MPI_Request *receiveRequests = NULL;
	List<CommBuffer*> *remoteReceives = getSortedList(true, remoteBuffers);
	int receiveCount = remoteReceives->NumElements();
	if (receiveCount > 0) {
		receiveRequests = issueAsyncReceives(remoteReceives);		
	}

	// wait for all receives to finish; see there is no writing back of data; this is because write will be invoked automatically
	if (receiveCount > 0) {
		int status = MPI_Waitall(receiveCount, receiveRequests, MPI_STATUSES_IGNORE);
		if (status != MPI_SUCCESS) {
			cout << "Segment "<< localSegmentTag << ": some of the asynchronous receives failed\n";
			exit(EXIT_FAILURE);
		}
	}

	delete remoteBuffers;
	delete remoteReceives;
	if (receiveCount > 0) delete[] receiveRequests;
	
	//*logFile << "\tCross-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
}

MPI_Request *CrossSyncCommunicator::issueAsyncReceives(List<CommBuffer*> *remoteReceiveBuffers) {

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int receiveCount = remoteReceiveBuffers->NumElements();
	MPI_Request *receiveRequests = new MPI_Request[receiveCount];

	for (int i = 0; i < remoteReceiveBuffers->NumElements(); i++) {
		CommBuffer *buffer = remoteReceiveBuffers->Nth(i);
		int bufferTag = buffer->getBufferTag();
		int bufferSize = buffer->getBufferSize();
		char *data = buffer->getData();
		int status = MPI_Irecv(data, bufferSize, MPI_CHAR, MPI_ANY_SOURCE, bufferTag, mpiComm, &receiveRequests[i]);
                if (status != MPI_SUCCESS) {
                	cout << "Segment " << localSegmentTag << ": could not issue asynchronous receive\n";
			exit(EXIT_FAILURE);
		}
		
	}
	return receiveRequests;
}

