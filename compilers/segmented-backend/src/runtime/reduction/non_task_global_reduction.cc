#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "reduction_barrier.h"
#include "non_task_global_reduction.h"
#include "../communication/mpi_group.h"

//-------------------------------------------------- Reduction Primitive -------------------------------------------------------

NonTaskGlobalReductionPrimitive::NonTaskGlobalReductionPrimitive(int elementSize, 
		ReductionOperator op, int localParticipants) 
		: NonTaskGlobalReductionBarrier(localParticipants) {
	
	this->elementSize = elementSize;
	this->op = op;
	this->logFile = NULL;
}

void NonTaskGlobalReductionPrimitive::updateLocalTarget(
		reduction::Result *finalResult, void *currLocalTarget) {

	if (op == MAX_ENTRY || op == MIN_ENTRY) {
                int *targetIndex = (int *) currLocalTarget;
                *targetIndex = finalResult->index;
        } else {
                memcpy(currLocalTarget, &(finalResult->data), elementSize);
        }
}

//------------------------------------------------ MPI Reduction Primitive -----------------------------------------------------

NonTaskGlobalMpiReductionPrimitive::NonTaskGlobalMpiReductionPrimitive(int elementSize,
		ReductionOperator op,
		int localParticipants,
		SegmentGroup *segmentGroup) 
		: NonTaskGlobalReductionPrimitive(elementSize, op, localParticipants) {

	this->segmentGroup = segmentGroup;

	// just declare sufficiently large buffers for participating in a reduction; they don't have to be
	// exactly as long as the data-type's size 
	sendBuffer = (char *) malloc(sizeof(char) * 50);
	receiveBuffer = (char *) malloc(sizeof(char) * 50);
}

void NonTaskGlobalMpiReductionPrimitive::releaseFunction() {

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
	NonTaskGlobalReductionPrimitive::releaseFunction();
}


