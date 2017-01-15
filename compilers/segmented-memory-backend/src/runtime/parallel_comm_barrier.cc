#include "parallel_comm_barrier.h"
#include "comm_barrier.h"
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <sys/time.h>
	
ParallelCommBarrier::ParallelCommBarrier(int size) {
	_size = size;
	_count=size;
        sem_init(&_mutex,0,1);                                   // init to 1
        _signalList = new List<SignalType>;
	_activeSignals = 0;
        _iterationNo = 0;
	pthread_barrier_init(&_barrier, NULL, _size);	
}

ParallelCommBarrier::~ParallelCommBarrier() {
	pthread_barrier_destroy(&_barrier);
	delete _signalList;
}

void ParallelCommBarrier::wait(SignalType signal, int iterationNo) {

	sem_wait(&_mutex);                                       // Make sure only one in at a  time

        // If there is no reason to wait then release the mutex and return
        if (!shouldWait(signal)) {
                sem_post(&_mutex);
                return;
        }

        _count--;                                               // decrease count
	
	int order = _count;					// the read value of the counter is
								// the order for any parallel processing
								// the thread will participate in						

        _signalList->Append(signal);                            // register the signal type

	if (_count == 0) {

                // Count the number of active signals
                _activeSignals = 0;
                for (int i = 0; i < _signalList->NumElements(); i++) {
                        SignalType signal = _signalList->Nth(i);
                        if (signal == REQUESTING_COMMUNICATION) {
                                _activeSignals++;
                        }
                }

		// join the barrier and kick off the before-transfer parallel processing
		struct timeval start;
        	gettimeofday(&start, NULL);
		pthread_barrier_wait(&_barrier);
		beforeTransfer(order, _size, _activeSignals);

		// wait on the barrier for all threads to finish before-transfer processing
		pthread_barrier_wait(&_barrier);
		struct timeval end;
        	gettimeofday(&end, NULL);
		recordTimingLog(BEFORE_TRANSFER_TIMING, start, end);

		// perform data transfer
		gettimeofday(&start, NULL);
		transferFunction(_activeSignals);
        	gettimeofday(&end, NULL);
		recordTimingLog(TRANSFER_TIMING, start, end);
								 
		// join the barrier again and kick of after-transfer parallel processing
		gettimeofday(&start, NULL);
		pthread_barrier_wait(&_barrier);
		afterTransfer(order, _size, _activeSignals);

                reset();                                        // Reset the barrier
                sem_post(&_mutex);                              // Release the mutex
		pthread_barrier_wait(&_barrier);		// release others by joining the barrier
        	
		gettimeofday(&end, NULL);
		recordTimingLog(AFTER_TRANSFER_TIMING, start, end);
        
	} else {
                sem_post(&_mutex);				// release the mutex first

		// wait on the barrier for the last thread to activate the before-transfer processing
		pthread_barrier_wait(&_barrier);

		// participate in the parallel before-transfer processing activity
		beforeTransfer(order, _size, _activeSignals);

		// wait on the barrier again to indicate that processing is done for the current thread
		pthread_barrier_wait(&_barrier);

		// wait again on the barrier for the last thread to complete data transfer so that 
		// after-transfer processing can be started
		pthread_barrier_wait(&_barrier);

		// participate in the parralel after-transfer processing activity
		afterTransfer(order, _size, _activeSignals);

		// lock ownself by waiting on the barrier one last time
		pthread_barrier_wait(&_barrier);
	}
}

void ParallelCommBarrier::reset() {
        _count = _size;                                         // Reset the counter
        _signalList->clear();                                   // Remove all signals
	_activeSignals = 0;					// Reset active signal count
        _iterationNo++; 					// Increase the iteration number
}

// By default always wait on the barrier
bool ParallelCommBarrier::shouldWait(SignalType signal) { return true; }

// There is no before transfer operation by default
void ParallelCommBarrier::beforeTransfer(int order, 
		int participants, int activeSignalsCount) {}

// There is no after transfer operation by default either
void ParallelCommBarrier::afterTransfer(int order, 
		int participants, int activeSignalsCount) {}

// By default timing log is not kept
void ParallelCommBarrier::recordTimingLog(TimingLogType logType, 
		struct timeval &start, struct timeval &end) {}


