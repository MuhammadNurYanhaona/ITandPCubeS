#ifndef _H_comm_barrier
#define _H_comm_barrier

/* This is an extension to Profe's barrier class for the use of communication control in segmented memory
   systems. Only one PPU (a.k.a a Pthread) performs the communication for a synchronization involving data
   transfer on behalf of everyone within a segment. Before the designated PPU can issue communication, all
   other relevant PPUs should finish their computation. Therefore, we needed a barrier. The barrier in the
   sync.h library is not enough as a provision for a release function, that will do a send or receive, is
   needed in this case. Furthermore, as IT communication are often contingent on whether or not a data 
   update has taken place within current segment. The wait() function needs a flag parameters that says if
   a particular PPU is just reporting that it has reached a certain point in its computation or if it is
   requesting a communication to take place. 
   This extension provides supports for these additional barrier features.  	
*/

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

#include "../../../../common-libs/utils/list.h"

// the two types for representing why a particular PPU is waiting on a communication barrier
enum SignalType {REQUESTING_COMMUNICATION, PASSIVE_REPORTING};

class CommBarrier {
  protected:
        int _size;			// How many threads need call wait before releasing all threads
        int _count;			// Waiting threads count at current instance
        sem_t mutex;    		// The mutex
        sem_t throttle; 		// Waiters signal the releaser so that there is no over-pumping
        sem_t waitq;    		// The semaphore on which the waiters wait	
	List<SignalType> *signalList;
	int iterationNo;		// How many times the barrier has been reset/reused so far
  public:
        CommBarrier(int size);
	virtual ~CommBarrier();
        void wait(SignalType signal, int iterationNo);
	void reset();

	// function to be extended by subclasses to make PPUs conditionally wait or bypass the barrier
	virtual bool shouldWait(SignalType signal, int iterationNo);

	// function to be extended by subclasses to execute appropriate communication logic
	virtual void releaseFunction(int activeSignalsCount);
};

#endif
