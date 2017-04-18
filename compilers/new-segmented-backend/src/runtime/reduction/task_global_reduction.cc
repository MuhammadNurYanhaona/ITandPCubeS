#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "reduction_barrier.h"
#include "task_global_reduction.h"
#include "../communication/mpi_group.h"

//-------------------------------------------------- Reduction Primitive -------------------------------------------------------

TaskGlobalReductionPrimitive::TaskGlobalReductionPrimitive(int elementSize, 
		ReductionOperator op, int localParticipants) 
		: TaskGlobalReductionBarrier(localParticipants) {
	
	this->elementSize = elementSize;
	this->op = op;
	this->intermediateResult = new reduction::Result();
	this->target = NULL;
	this->logFile = NULL;
}

void TaskGlobalReductionPrimitive::initFunction(reduction::Result *localPartialResult, void *target) { 

	// save the target address
	this->target = target;					
	
	// copy the first PPU's result to the intermediate result
	memcpy(intermediateResult, localPartialResult, sizeof(reduction::Result));	
}

void TaskGlobalReductionPrimitive::releaseFunction() {
	if (op == MAX_ENTRY || op == MIN_ENTRY) {
		int *targetIndex = (int *) target;
		*targetIndex = intermediateResult->index;
	} else {
		memcpy(target, &(intermediateResult->data), elementSize);
	}
}

//------------------------------------------------ MPI Reduction Primitive -----------------------------------------------------

TaskGlobalMpiReductionPrimitive::TaskGlobalMpiReductionPrimitive(int elementSize,
		ReductionOperator op,
		int localParticipants,
		SegmentGroup *segmentGroup) 
		: TaskGlobalReductionPrimitive(elementSize, op, localParticipants) {

	this->segmentGroup = segmentGroup;

	// just declare sufficiently large buffers for participating in a reduction; they don't have to be
	// exactly as long as the data-type's size 
	sendBuffer = (char *) malloc(sizeof(char) * 50);
	receiveBuffer = (char *) malloc(sizeof(char) * 50);
}

void TaskGlobalMpiReductionPrimitive::releaseFunction() {

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
	TaskGlobalReductionPrimitive::releaseFunction();
}


