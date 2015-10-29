#include "../utils/binary_search.h"
#include "../utils/utility.h"
#include "scalar_communicator.h"
#include "communicator.h"
#include <vector>
#include <iostream>

using namespace std;

//----------------------------------------------------------- Scalar Communicator -------------------------------------------------------/

ScalarCommunicator::ScalarCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags, 
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: Communicator(localSegmentTag, dependencyName, 
			localSenderPpus, localReceiverPpus) {

	this->dataSize = dataSize;
	this->dataBuffer = NULL;
	this->senderSegmentTags = senderSegmentTags;
	this->receiverSegmentTags = receiverSegmentTags;
	
	std::vector<int> *participants = getParticipantsTags();
	active = participants->size() > 1;
	delete participants;
}

std::vector<int> *ScalarCommunicator::getParticipantsTags() {
	std::vector<int> *participants = new std::vector<int>(*senderSegmentTags);
	for (int i = 0; i < receiverSegmentTags->size(); i++) {
		int tag = receiverSegmentTags->at(i);
		binsearch::insertIfNotExist(participants, tag);
	}
	return participants;
}

//----------------------------------------------------- Scalar Replica Sync Communicator ------------------------------------------------/

ScalarReplicaSyncCommunicator::ScalarReplicaSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags, 
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			senderSegmentTags, 
			receiverSegmentTags,
			localSenderPpus, localReceiverPpus, 
			dataSize) {}

void ScalarReplicaSyncCommunicator::send() {
	
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
	
	status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, myRank, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "Segment " << localSegmentTag << ": could not broadcast update to scalar variable\n";
		exit(EXIT_FAILURE);
	}
}

void ScalarReplicaSyncCommunicator::receive() {
	
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
		status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": could not receive broadcast\n";
			exit(EXIT_FAILURE);
		}
	}
}

//-------------------------------------------------------- Scalar Up Sync Communicator --------------------------------------------------/

ScalarUpSyncCommunicator::ScalarUpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags,
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			senderSegmentTags, 
			receiverSegmentTags,
			localSenderPpus, localReceiverPpus, 
			dataSize) {

	Assert(receiverSegmentTags->size() == 1);
}

void ScalarUpSyncCommunicator::send() {
	
	int receiverSegment = receiverSegmentTags->at(0);
	if (receiverSegment == localSegmentTag) return;

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int receiver = segmentGroup->getRank(receiverSegment);
	int status = MPI_Send(dataBuffer, dataSize, MPI_CHAR, receiver, 0, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not send update to upper level segment\n";
		exit(EXIT_FAILURE);
	}	
}

void ScalarUpSyncCommunicator::receive() {
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int status = MPI_Recv(dataBuffer, dataSize, MPI_CHAR, MPI_ANY_SOURCE, 0, mpiComm, MPI_STATUS_IGNORE);
	if (status != MPI_SUCCESS) {
		cout << "could not receive update message from unknown sub-source\n";
		exit(EXIT_FAILURE);
	}
}

//------------------------------------------------------- Scalar Down Sync Communicator -------------------------------------------------/

ScalarDownSyncCommunicator::ScalarDownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags,
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			senderSegmentTags, 
			receiverSegmentTags,
			localSenderPpus, localReceiverPpus, 
			dataSize) {

	Assert(senderSegmentTags->size() == 1);
}

void ScalarDownSyncCommunicator::send() {
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int rank = segmentGroup->getRank(localSegmentTag);
	int status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, rank, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not broadcast update from upper level LPS\n";
		exit(EXIT_FAILURE);
	}
}
        
void ScalarDownSyncCommunicator::receive() {
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int broadcastingSegment = senderSegmentTags->at(0);
	int broadcaster = segmentGroup->getRank(broadcastingSegment);
	int status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, broadcaster, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not receive broadcast from upper level LPS\n";
		exit(EXIT_FAILURE);
	}
}

