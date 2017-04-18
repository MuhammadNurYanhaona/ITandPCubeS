#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "reduction_barrier.h"
#include "../../../../common-libs/utils/list.h"

//---------------------------------------------- Task Global Reduction Barrier -------------------------------------------------

TaskGlobalReductionBarrier::TaskGlobalReductionBarrier(int size) {
        _size=size;
        _count=size;
        sem_init(&mutex,0,1);   				// init to 1
        sem_init(&throttle,0,0);        			// init to 0
        sem_init(&waitq,0,0);   				// init to 0
}

void TaskGlobalReductionBarrier::reduce(reduction::Result *localPartialResult, void *target) {

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

//--------------------------------------------- Non Task Global Reduction Barrier ----------------------------------------------

NonTaskGlobalReductionBarrier::NonTaskGlobalReductionBarrier(int size) {
        _size=size;
        _count=size;
        sem_init(&mutex,0,1);   				// init to 1
        sem_init(&throttle,0,0);        			// init to 0
        sem_init(&waitq,0,0);   				// init to 0
	localTargets = new List<void*>;				// empty list
	intermediateResult = NULL;				// NULL reference
}

void NonTaskGlobalReductionBarrier::reduce(reduction::Result *localPartialResult,
		void *localTarget,
		reduction::Result *toBeStoredFinalResult) {
	
	sem_wait(&mutex);					// Make sure only one in at a  time
	if (_count == _size) {
		initFunction(localPartialResult, 		// Do any initialization needed at the first PPU controller's
				toBeStoredFinalResult);		// barrier entrance 
	} else {
		entryFunction(localPartialResult);		// Update the partial result of intra-segment reduction based 
								// on caller PPU controller's computed value 
	}
	_count--;						// update the counter
	localTargets->Append(localTarget);			// hold the current PPU controller's local target reference

	if (_count == 0) {
		
		executeFinalStepOfReduction();			// execute the final step to do any cross-segment operation at 
								// the end, if needed, and a cleanup.
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

void NonTaskGlobalReductionBarrier::initFunction(reduction::Result *localPartialResult,
		reduction::Result *toBeStoredFinalResult) {

	// copy the first PPU's result to the storage result
        memcpy(toBeStoredFinalResult, localPartialResult, sizeof(reduction::Result));

	// grasp the reference of the storage result
	this->intermediateResult = toBeStoredFinalResult;
}

void NonTaskGlobalReductionBarrier::updateAllLocalTargets() {
	
	for (int i = 0; i < localTargets->NumElements(); i++) {
		void *currLocalTarget = localTargets->Nth(i);
		updateLocalTarget(intermediateResult, currLocalTarget);
	}
}

void NonTaskGlobalReductionBarrier::executeFinalStepOfReduction() {

	// execute the release function to let the subclass do necessary processing of the final step 
	releaseFunction();

	// update all local targets of different PPU controllers
	updateAllLocalTargets();

	// we need to reset the reference for final result storage; as a subsequent use of the barrier 
	// is supposed set up a new reference
	intermediateResult = NULL;

	// it is important to clear the list after the updates to reset the barrier for next-time use
	localTargets->clear();
}
