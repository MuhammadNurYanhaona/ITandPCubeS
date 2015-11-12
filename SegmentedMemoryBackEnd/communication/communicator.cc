#include "../utils/list.h"
#include "../runtime/comm_barrier.h"
#include "comm_buffer.h"
#include "communicator.h"
#include <iostream>
#include <cstdlib>
#include <sstream>

//-------------------------------------------------------------- Send Barrier ------------------------------------------------------------/

SendBarrier::SendBarrier(int participantCount, Communicator *communicator) : CommBarrier(participantCount) {
	this->communicator = communicator;
}

bool SendBarrier::shouldWait(SignalType signal, int iterationNo) {
	return communicator->shouldWaitOnSend(signal, iterationNo);
}

void SendBarrier::releaseFunction(int activeSignalsCount) {
	if (communicator->shouldSend(activeSignalsCount)) {
		executeSend();
	}
}

void SendBarrier::executeSend() {
	communicator->prepareBuffersForSend();
	communicator->sendData();
	communicator->afterSend();
}

//------------------------------------------------------------- Receive Barrier ----------------------------------------------------------/

ReceiveBarrier::ReceiveBarrier(int participantCount, Communicator *communicator) : CommBarrier(participantCount) {
	this->communicator = communicator;
}

bool ReceiveBarrier::shouldWait(SignalType signal, int iterationNo) {
	return communicator->shouldWaitOnReceive(signal, iterationNo);
}

void ReceiveBarrier::releaseFunction(int activeSignalsCount) {
	if (communicator->shouldReceive(activeSignalsCount)) {
		executeReceive();
	}
}

void ReceiveBarrier::executeReceive() {
	communicator->receiveData();
	communicator->processBuffersAfterReceive();
	communicator->afterReceive();
}

//-------------------------------------------------------------- Communicator ------------------------------------------------------------/

Communicator::Communicator(int localSegmentTag, 
		const char *dependencyName, 
		int localSenderPpus, int localReceiverPpus) : CommBufferManager(dependencyName) {
	
	this->localSegmentTag = localSegmentTag;

	if (localSenderPpus > 0) {
		sendBarrier = new SendBarrier(localSenderPpus, this);
	} else {
		sendBarrier = NULL;
	}
	if (localReceiverPpus > 0) {
		receiveBarrier = new ReceiveBarrier(localReceiverPpus, this);
	} else {
		receiveBarrier = NULL;
	}
	iterationNo = 0;
	communicatorId = 0;
}

void Communicator::setupBufferTags(int communicatorId, int totalSegmentsInMachine) {
	this->communicatorId = communicatorId;
	std::ostringstream digitStr;
	digitStr << totalSegmentsInMachine;
	int digitsForSegmentId = digitStr.str().length();
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		CommBuffer *buffer = commBufferList->Nth(i);
		buffer->setBufferTag(communicatorId, digitsForSegmentId);
	}
}

void Communicator::setupCommunicator(bool includeNonInteractingSegments) {
	*logFile << "\tSetting up communicator for " << dependencyName << "\n";
	logFile->flush();
	if (includeNonInteractingSegments) {
        	segmentGroup = new SegmentGroup(*participantSegments);
	} else {
		std::vector<int> *interactingParticipants = getParticipantsTags();
        	segmentGroup = new SegmentGroup(*interactingParticipants);
		delete interactingParticipants;
	}
        segmentGroup->setupCommunicator(*logFile);
	*logFile << "\tSetup done for communicator for " << dependencyName << "\n";
	logFile->flush();
}

