#include "../utils/binary_search.h"
#include "../utils/utility.h"
#include "scalar_communicator.h"
#include "communicator.h"
#include <vector>

//----------------------------------------------------------- Scalar Communicator -------------------------------------------------------/

ScalarCommunicator::ScalarCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags, int dataSize) 
		: Communicator(localSegmentTag, dependencyName, 
			senderSegmentTags->size(), receiverSegmentTags->size()) {

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
                std::vector<int> *receiverSegmentTags, int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			senderSegmentTags, receiverSegmentTags,
			dataSize) {}

void ScalarReplicaSyncCommunicator::participate(bool sending) {}

//-------------------------------------------------------- Scalar Up Sync Communicator --------------------------------------------------/

ScalarUpSyncCommunicator::ScalarUpSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags, int dataSize)
		: ScalarCommunicator(localSegmentTag, dependencyName, 
                        senderSegmentTags, receiverSegmentTags,
                        dataSize) {
	Assert(receiverSegmentTags->size() == 0);
}

void ScalarUpSyncCommunicator::send() {}

void ScalarUpSyncCommunicator::receive() {}

//------------------------------------------------------- Scalar Down Sync Communicator -------------------------------------------------/

ScalarDownSyncCommunicator::ScalarDownSyncCommunicator(int localSegmentTag,
                const char *dependencyName,
                std::vector<int> *senderSegmentTags,
                std::vector<int> *receiverSegmentTags, int dataSize) 
		: ScalarCommunicator(localSegmentTag, dependencyName, 
			senderSegmentTags, receiverSegmentTags,
			dataSize) {
	Assert(senderSegmentTags->size() == 0);
}

void ScalarDownSyncCommunicator::send() {}
        
void ScalarDownSyncCommunicator::receive() {}

