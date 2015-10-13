#include "../utils/list.h"
#include "../runtime/comm_barrier.h"
#include "comm_buffer.h"
#include "communicator.h"
#include <iostream>
#include <cstdlib>

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

Communicator::Communicator(const char *dependencyName, int senderCount, int receiverCount) 
		: CommBufferManager(dependencyName) {
	if (senderCount > 0) {
		sendBarrier = new SendBarrier(senderCount, this);
	} else {
		sendBarrier = NULL;
	}
	if (receiverCount > 0) {
		receiveBarrier = new ReceiveBarrier(receiverCount, this);
	} else {
		receiveBarrier = NULL;
	}
	iterationNo = 0;
}
