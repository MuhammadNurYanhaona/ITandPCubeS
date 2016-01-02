#include "../utils/list.h"
#include "../runtime/comm_barrier.h"
#include "comm_buffer.h"
#include "comm_statistics.h"
#include "communicator.h"
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <time.h>
#include <sys/time.h>

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
	struct timeval start;
        gettimeofday(&start, NULL);
	communicator->prepareBuffersForSend();
	communicator->sendData();
	communicator->afterSend();
	struct timeval end;
        gettimeofday(&end, NULL);
	CommStatistics *commStat = communicator->getCommStat();
	commStat->addCommunicationTime(communicator->getName(), start, end);
}

//------------------------------------------------------------- Receive Barrier ----------------------------------------------------------/

ReceiveBarrier::ReceiveBarrier(int participantCount, Communicator *communicator) : CommBarrier(participantCount) {
	this->communicator = communicator;
}

bool ReceiveBarrier::shouldWait(SignalType signal, int iterationNo) {
	return communicator->shouldWaitOnReceive(signal, iterationNo);
}

void ReceiveBarrier::releaseFunction(int activeSignalsCount) {
	if (communicator->shouldReceive(activeSignalsCount, iterationNo)) {
		executeReceive();
	}
}

void ReceiveBarrier::executeReceive() {
	struct timeval start;
        gettimeofday(&start, NULL);
	communicator->receiveData();
	communicator->processBuffersAfterReceive();
	communicator->afterReceive();
	struct timeval end;
        gettimeofday(&end, NULL);
	CommStatistics *commStat = communicator->getCommStat();
	commStat->addCommunicationTime(communicator->getName(), start, end);
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
	commStat = NULL;
}

void Communicator::describe(int indentation) {
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	*logFile << indent.str() << "Communicator for " << dependencyName << ":\n";
	*logFile << indent.str() << '\t' << "Total Communication Buffers: ";
	*logFile << commBufferList->NumElements() << "\n";
	for (int i = 0; i < commBufferList->NumElements(); i++) {
		*logFile << indent.str() << "\tBuffer #" << i + 1 << ":\n";
		CommBuffer *buffer = commBufferList->Nth(i);
		buffer->describe(*logFile, indentation + 1);
	}
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
	
	struct timeval start;
        gettimeofday(&start, NULL);
	if (includeNonInteractingSegments) {
        	segmentGroup = new SegmentGroup(*participantSegments);
	} else {
		std::vector<int> *interactingParticipants = getParticipantsTags();
        	segmentGroup = new SegmentGroup(*interactingParticipants);
		delete interactingParticipants;
	}
        segmentGroup->setupCommunicator(*logFile);
	struct timeval end;
        gettimeofday(&end, NULL);
	commStat->addCommResourcesSetupTime(dependencyName, start, end);

	*logFile << "\tSetup done for communicator for " << dependencyName << "\n";
	logFile->flush();
}

void Communicator::excludeOwnselfFromCommunication(const char *dependencyName, 
		int localSegmentTag, std::ofstream &logFile) {
	logFile << "\tExcluding myself from dependency " << dependencyName << "\n";
	logFile.flush();
	SegmentGroup::excludeSegmentFromGroupSetup(localSegmentTag, logFile);
	logFile << "\tExcluded myself from dependency " << dependencyName << "\n";
	logFile.flush();
}

