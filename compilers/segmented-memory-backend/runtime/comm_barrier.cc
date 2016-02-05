#include "comm_barrier.h"
#include "../utils/list.h"
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

CommBarrier::CommBarrier(int size) {
        _size=size;
        _count=size;
        sem_init(&mutex,0,1);           			// init to 1
        sem_init(&throttle,0,0);        			// init to 0
        sem_init(&waitq,0,0);           			// init to 0
	signalList = new List<SignalType>;
	iterationNo = 0;
}

void CommBarrier::wait(SignalType signal, int iterationNo) {

        sem_wait(&mutex);       				// Make sure only one in at a  time
	
	// If there is no reason to wait then release the mutex and return
	if (!shouldWait(signal, iterationNo)) {
                sem_post(&mutex);
		return;	
	}

        _count--;						// decrease count
	signalList->Append(signal);				// register the signal type

	// Barrier release situation
        if (_count == 0) {
                // Count the number of active signals
		int activeSignals = 0;
		for (int i = 0; i < signalList->NumElements(); i++) {
			SignalType signal = signalList->Nth(i);
			if (signal == REQUESTING_COMMUNICATION) {
				activeSignals++;
			}
		}
		
		releaseFunction(activeSignals);			// Execute barrier release routine

		// Time to wake everyone up
                for (int i=1;i<_size;i++) {
                        sem_post(&waitq);       		// Wake up another waiter
                        sem_wait(&throttle);    		// Wait for the waiter to awaken and signal me.
                }

                reset();           				// Reset the barrier
                sem_post(&mutex);       			// Release the mutex
        }
	// Waiting situation
        else {
                // Block myself, but first release mutex
                sem_post(&mutex);
                sem_wait(&waitq);       			// Sleep
                sem_post(&throttle);    			// wake up the releaser
        }
}

CommBarrier::~CommBarrier() { delete signalList; }

void CommBarrier::reset() { 
	_count = _size;						// Reset the counter
	signalList->clear(); 					// Remove all signals
	iterationNo++;						// increase the iteration number
}

bool CommBarrier::shouldWait(SignalType signal, int iterationNo) { 
	return true;						// By default always wait on the barrier
}

void CommBarrier::releaseFunction(int activeSignalsCount) {}	// Default implementation does nothing
