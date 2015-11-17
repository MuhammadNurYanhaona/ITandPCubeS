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
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: Communicator(localSegmentTag, dependencyName, 
			localSenderPpus, localReceiverPpus) {
	this->dataSize = dataSize;
	this->dataBuffer = NULL;
	this->hasLocalSender = localSenderPpus > 0;
	this->hasLocalReceiver = localReceiverPpus > 0;
	this->active = false;
}

void ScalarCommunicator::setupCommunicator(bool includeNonInteractingSegments) {
	
	*logFile << "\tSetting up scalar communicator for " << dependencyName << "\n";
	logFile->flush();

	segmentGroup = new SegmentGroup();
	segmentGroup->discoverGroupAndSetupCommunicator(*logFile);
	int participants = segmentGroup->getParticipantsCount();

	if (participants > 1) {
		this->active = true;
		
		int mySendReceiveConfig[2];
		mySendReceiveConfig[0] = hasLocalSender ? 1 : 0;
		mySendReceiveConfig[1] = hasLocalReceiver ? 1 : 0;
		
		int myRank = segmentGroup->getRank(localSegmentTag);
		MPI_Comm mpiComm = segmentGroup->getCommunicator();
		int *sendReceiveConfigs = new int[participants * 2];
		int status = MPI_Allgather(mySendReceiveConfig, 2, MPI_INT, sendReceiveConfigs, 2, MPI_INT, mpiComm);
		if (status != MPI_SUCCESS) {
			*logFile << "Could not determine who will be sending and receiving updates\n";
			logFile->flush();
			exit(EXIT_FAILURE);	
		}
		for (int i = 0; i < participants; i++) {
			int segmentTag = segmentGroup->getSegment(i);
			if (sendReceiveConfigs[i * 2] == 1) {
				senderSegmentTags.push_back(segmentTag);
			}
			if (sendReceiveConfigs[i * 2 + 1] == 1) {
				receiverSegmentTags.push_back(segmentTag);
			}
		}
	}
	
	*logFile << "\tSetup done for scalar communicator " << dependencyName << "\n";
	logFile->flush();
}

//----------------------------------------------------- Scalar Replica Sync Communicator ------------------------------------------------/

ScalarReplicaSyncCommunicator::ScalarReplicaSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			localSenderPpus, localReceiverPpus, 
			dataSize) {}

void ScalarReplicaSyncCommunicator::send() {
	
	//*logFile << "\tScalar replica communicator is sending data for " << dependencyName << "\n";
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
	
	status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, myRank, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "Segment " << localSegmentTag << ": could not broadcast update to scalar variable\n";
		exit(EXIT_FAILURE);
	}
	
	//*logFile << "\tScalar replica communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}

void ScalarReplicaSyncCommunicator::receive() {
	
	//*logFile << "\tScalar replica communicator is waiting for data for " << dependencyName << "\n";
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
		status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, broadcaster, mpiComm);
		if (status != MPI_SUCCESS) {
			cout << "Segment " << localSegmentTag << ": could not receive broadcast\n";
			exit(EXIT_FAILURE);
		}
	}

	//*logFile << "\tScalar replica communicator received data for " << dependencyName << "\n";
	//logFile->flush();
}

//-------------------------------------------------------- Scalar Up Sync Communicator --------------------------------------------------/

ScalarUpSyncCommunicator::ScalarUpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			localSenderPpus, localReceiverPpus, 
			dataSize) {}

void ScalarUpSyncCommunicator::send() {
	
	//*logFile << "\tScalar up-sync communicator is sending data for " << dependencyName << "\n";
	//logFile->flush();

	int receiverSegment = receiverSegmentTags.at(0);
	if (receiverSegment == localSegmentTag) return;

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int receiver = segmentGroup->getRank(receiverSegment);
	int status = MPI_Send(dataBuffer, dataSize, MPI_CHAR, receiver, 0, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not send update to upper level segment\n";
		exit(EXIT_FAILURE);
	}
	
	//*logFile << "\tScalar up-sync communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}

void ScalarUpSyncCommunicator::receive() {

	//*logFile << "\tScalar up-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();
	
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int status = MPI_Recv(dataBuffer, dataSize, MPI_CHAR, MPI_ANY_SOURCE, 0, mpiComm, MPI_STATUS_IGNORE);
	if (status != MPI_SUCCESS) {
		cout << "could not receive update message from unknown sub-source\n";
		exit(EXIT_FAILURE);
	}

	//*logFile << "\tScalar up-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
}

//------------------------------------------------------- Scalar Down Sync Communicator -------------------------------------------------/

ScalarDownSyncCommunicator::ScalarDownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
		int localSenderPpus,
                int localReceiverPpus,
		int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			localSenderPpus, localReceiverPpus, 
			dataSize) {}

void ScalarDownSyncCommunicator::send() {
	
	//*logFile << "\tScalar down-sync communicator is sending data for " << dependencyName << "\n";
	//logFile->flush();

	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int rank = segmentGroup->getRank(localSegmentTag);
	int status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, rank, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not broadcast update from upper level LPS\n";
		exit(EXIT_FAILURE);
	}

	//*logFile << "\tScalar down-sync communicator sent data for " << dependencyName << "\n";
	//logFile->flush();
}
        
void ScalarDownSyncCommunicator::receive() {
	
	//*logFile << "\tScalar down-sync communicator is waiting for data for " << dependencyName << "\n";
	//logFile->flush();
	
	MPI_Comm mpiComm = segmentGroup->getCommunicator();
	int broadcastingSegment = senderSegmentTags.at(0);
	int broadcaster = segmentGroup->getRank(broadcastingSegment);
	int status = MPI_Bcast(dataBuffer, dataSize, MPI_CHAR, broadcaster, mpiComm);
	if (status != MPI_SUCCESS) {
		cout << "could not receive broadcast from upper level LPS\n";
		exit(EXIT_FAILURE);
	}
	
	//*logFile << "\tScalar down-sync communicator received data for " << dependencyName << "\n";
	//logFile->flush();
}

