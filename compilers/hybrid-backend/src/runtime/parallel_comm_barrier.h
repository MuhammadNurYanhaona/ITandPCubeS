#ifndef _H_parallel_comm_barrier
#define _H_parallel_comm_barrier

/* This parallel-comm-barrier is an extension to the comm-barrier class in CommBarrier.h. This extension has
 * been added to take benefit of multiple thread participants to have parallel implementations of the parts of
 * a dependency resolution process that can be parallelized. Note that a dependency resolution involving data
 * movements within and/or across segments have three steps: first, read data from the part-container-tree 
 * into one or more communication buffers; second, do MPI communication; finally, write any data received to
 * proper parts in the receiver side part-container tree. Among these three broad operations, the first and 
 * the last are parallelizable. This extension allows subclasses to provide parallel implementations for those
 * two steps that are called at proper time during the waiting process.
 */

#include "comm_barrier.h"
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <sys/time.h>

// an type list to allow recording of time spent on specific communication related activity on the barrier
enum TimingLogType { BEFORE_TRANSFER_TIMING, TRANSFER_TIMING, AFTER_TRANSFER_TIMING };

class ParallelCommBarrier {
  protected:
        int _size;                      // How many threads need call wait before releasing all threads
        int _count;                     // Waiting threads count at current instance
        sem_t _mutex;                   // The mutex protecting access to elements of the barrier
        List<SignalType> *_signalList;	// Collections of reasons the participants have contacted this barrier
	int _activeSignals;		// How many of the received signals requesting a communication
        int _iterationNo;               // How many times the barrier has been reset/reused so far
	pthread_barrier_t _barrier;	// Internal barrier needed for stepping through different phases
  public:	
	ParallelCommBarrier(int size);
        virtual ~ParallelCommBarrier();

	// function to be invoked from outside; it embodies the entire workflow for communications; 
	// Note the second parameter. The user of the barrier is supposed to keep track of the number of 
	// times the barrier has been used. The user can advance the iteration no or hold it unchanged in 
	// specific scenarios to skip communications or do the same communication multiple times.
        void wait(SignalType signal, int callerIterationNo);

	// function to reset the barrier for any subsequent use
        void reset();

	// function to be extended by subclasses to make PPUs conditionally wait or bypass the barrier
        virtual bool shouldWait(SignalType signal, int callerIterationNo);

	// function to determine whether or not to skip data transfer in a specific scenario
	virtual bool shouldPerformTransfer(int activeSignalsCount, int callerIterationNo) = 0;

	// function to be extended by subclasses to distribute any communication buffer preparation and other
	// parallelizable activities that precede actual communications. Different participant threads should
	// get a different value for the order attribute so that the subclass implementation of the function 
	// can properly distribute work.
	virtual void beforeTransfer(int order, int participants);

	// function to be extended by subclasses to do the data transfer
	virtual void transferFunction() = 0;

	// function to be extended by subclasses to distribute any parallelizable post processing step
	virtual void afterTransfer(int order, int participants);

	// logging function to be utilized by subclasses to record communication performance
	virtual void recordTimingLog(TimingLogType logType, struct timeval &start, struct timeval &end);
};

#endif
