#include "comm_barrier.h"
#include "parallel_comm_barrier.h"

#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
	
ParallelCommBarrier::ParallelCommBarrier(int size) {
	_size = size;
	_count=size;
        sem_init(&_mutex,0,1);                                   	// initialized to 1
        _signalList = new List<SignalType>;
	_activeSignals = 0;
        _iterationNo = 0;
	pthread_barrier_init(&_barrier, NULL, _size);	
}

ParallelCommBarrier::~ParallelCommBarrier() {
	pthread_barrier_destroy(&_barrier);
	delete _signalList;
}

void ParallelCommBarrier::wait(SignalType signal, int callerIterationNo) {

	sem_wait(&_mutex);                                       	// Make sure only one is in at a time

	// If there is no reason to wait then release the mutex and return
        if (!shouldWait(signal, callerIterationNo)) {
                sem_post(&_mutex);
                return;
        }

        _count--;                                               	// decrease count
	
	int order = _count;						// the read value of the counter is
									// the order for any parallel processing
									// the thread will participate in						

        _signalList->Append(signal);                            	// register the signal type

	if (_count == 0) {

                // Count the number of active signals
                _activeSignals = 0;
                for (int i = 0; i < _signalList->NumElements(); i++) {
                        SignalType signal = _signalList->Nth(i);
                        if (signal == REQUESTING_COMMUNICATION) {
                                _activeSignals++;
                        }
                }

		// join the barrier to let all participants determine if a transfer should take place 
		pthread_barrier_wait(&_barrier);
		if (shouldPerformTransfer(_activeSignals, callerIterationNo)) {
			
			// kick off the before-transfer parallel processing
			struct timeval start;
			gettimeofday(&start, NULL);
			beforeTransfer(order, _size);

			// wait on the barrier for all threads to finish before-transfer processing
			pthread_barrier_wait(&_barrier);
			struct timeval end;
			gettimeofday(&end, NULL);
			recordTimingLog(BEFORE_TRANSFER_TIMING, start, end);

			// perform data transfer
			gettimeofday(&start, NULL);
			transferFunction();
			gettimeofday(&end, NULL);
			recordTimingLog(TRANSFER_TIMING, start, end);
									 
			// join the barrier again and kick of after-transfer parallel processing
			gettimeofday(&start, NULL);
			pthread_barrier_wait(&_barrier);
			afterTransfer(order, _size);

			reset();                                        // Reset the barrier
			pthread_barrier_wait(&_barrier);		// release others by joining the barrier
			
			gettimeofday(&end, NULL);
			recordTimingLog(AFTER_TRANSFER_TIMING, start, end);
			sem_post(&_mutex);                              // Release the mutex
		} else {
			// reset the barrier for subsequent iterations
			reset();                                        // Reset the barrier
			pthread_barrier_wait(&_barrier);		// release others by joining the barrier
			sem_post(&_mutex);                              // Release the mutex
		}
    
	} else {
                sem_post(&_mutex);				// release the mutex first

		// wait on the barrier for the last thread to count active signals to check the need of a 
		// data transfer
		pthread_barrier_wait(&_barrier);
		if (shouldPerformTransfer(_activeSignals, callerIterationNo)) {

			// participate in the parallel before-transfer processing activity
			beforeTransfer(order, _size);

			// wait on the barrier again to indicate that processing is done for the current thread
			pthread_barrier_wait(&_barrier);

			// wait again on the barrier for the last thread to complete data transfer so that 
			// after-transfer processing can be started
			pthread_barrier_wait(&_barrier);

			// participate in the parralel after-transfer processing activity
			afterTransfer(order, _size);

			// lock ownself by waiting on the barrier one last time
			pthread_barrier_wait(&_barrier);
		} else {
			// wait for the barrier reset before leaving
			pthread_barrier_wait(&_barrier);
		}
	}
}

void ParallelCommBarrier::reset() {
        _count = _size;                                         // Reset the counter
        _signalList->clear();                                   // Remove all signals
	_activeSignals = 0;					// Reset active signal count
        _iterationNo++; 					// Increase the iteration number
}

// By default always wait on the barrier
bool ParallelCommBarrier::shouldWait(SignalType signal, int callerIterationNo) { return true; }

// There is no before transfer operation by default
void ParallelCommBarrier::beforeTransfer(int order, int participants) {}

// There is no after transfer operation by default either
void ParallelCommBarrier::afterTransfer(int order, int participants) {}

// By default timing log is not kept
void ParallelCommBarrier::recordTimingLog(TimingLogType logType, 
		struct timeval &start, struct timeval &end) {}


