#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "reduction.h"

//--------------------------------------------------- Reduction Result ---------------------------------------------------------



//--------------------------------------------------- Reduction Barrier --------------------------------------------------------

ReductionBarrier::ReductionBarrier(int size) {
        _size=size;
        _count=size;
        sem_init(&mutex,0,1);   				// init to 1
        sem_init(&throttle,0,0);        			// init to 0
        sem_init(&waitq,0,0);   				// init to 0
}

void ReductionBarrier::reduce(reduction::Result *localPartialResult, void *target) {

	sem_wait(&mutex);					// Make sure only one in at a  time
	if (_count == _size) {
		initFunction(localPartialResult, target);	// Do any initialization needed at the first PPU controller's 
								// barrier entrance
	} else {
		entryFunction(localPartialResult);		// Update the partial result of intra-segment reduction based 
	}							// on caller PPU controller's computed value 
	
	_count--;						// update the counter

	if (_count == 0) {
		
		releaseFunction();				// Do any cross-segment operation at the end, if needed.
		
		// Time to wake everyone up
		for (int i = 1; i<_size; i++) {
			sem_post(&waitq);			// Wake up another waiter
			sem_wait(&throttle);			// Wait for the waiter to awaken and signal me.
		}
		_count=_size;					// Reset the counter
		sem_post(&mutex);				// Release the mutex
	}
	else {
		sem_post(&mutex);				// Block myself, but first release mutex
		sem_wait(&waitq);				// Sleep
		sem_post(&throttle);				// Wake up the releaser
	}
}

//-------------------------------------------------- Reduction Primitive -------------------------------------------------------

ReductionPrimitive::ReductionPrimitive(int elementSize, ReductionOperator op, int localParticipants) 
		: ReductionBarrier(localParticipants) {
	this->elementSize = elementSize;
	this->op = op;
	this->intermediateResult = new reduction::Result();
	this->target = NULL;
	this->logFile = NULL;
}

void ReductionPrimitive::initFunction(reduction::Result *localPartialResult, void *target) { 

	// save the target address
	this->target = target;					
	
	// copy the first PPU's result to the intermediate result
	memcpy(intermediateResult, localPartialResult, sizeof(reduction::Result));	
}

void ReductionPrimitive::releaseFunction() {
	if (op == MAX_ENTRY || op == MIN_ENTRY) {
		int *targetIndex = (int *) target;
		*targetIndex = intermediateResult->index;
	} else {
		memcpy(target, &(intermediateResult->data), elementSize);
	}
}

//------------------------------------------------ MPI Reduction Primitive -----------------------------------------------------

MpiReductionPrimitive::MpiReductionPrimitive(int elementSize,
		ReductionOperator op,
		int localParticipants,
		SegmentGroup *segmentGroup) 
		: ReductionPrimitive(elementSize, op, localParticipants) {

	this->segmentGroup = segmentGroup;

	// just declare sufficiently large buffers for participating in a reduction; they don't have to be
	// exactly as long as the data-type's size 
	sendBuffer = (char *) malloc(sizeof(char) * 50);
	receiveBuffer = (char *) malloc(sizeof(char) * 50);
}

void MpiReductionPrimitive::releaseFunction() {

	if (segmentGroup != NULL) {
	
		// copy data into the send buffer
		memcpy(sendBuffer, &(intermediateResult->data), elementSize);
	
		// copy index next to the data
		char *sendIndex = sendBuffer + elementSize;
		memcpy(sendIndex, &(intermediateResult->index), sizeof(unsigned int));  
	
		// do MPI communication as needed
		performCrossSegmentReduction();
		
		// copy data from the receive buffer
		memcpy(&(intermediateResult->data), receiveBuffer, elementSize);
	
		// copy index from next to data position of the receive buffer
		char *receiveIndex = receiveBuffer + elementSize;
		memcpy(&(intermediateResult->index), receiveIndex, sizeof(unsigned int));
	}  
	
	// then call super-class's release function to copy the result to the target
	ReductionPrimitive::releaseFunction();
}


