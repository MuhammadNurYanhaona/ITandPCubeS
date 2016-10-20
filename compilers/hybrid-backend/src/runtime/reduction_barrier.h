#ifndef _H_reduction_barrier
#define _H_reduction_barrier

/* This is another extension of Profe's barrier class. It is designed for implementing segment-local and cross-
   segment reductions. It provides three plug points to insert custom, context dependent logic inside the synchro-
   nization process.	    
*/

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include "reduction.h"

class ReductionBarrier {
  private:
	int _size;				// How many threads need call wait before releasing all threads 
	int _count;				// Current count of waiting threads
	sem_t mutex;  				// The mutex
	sem_t throttle;				// Waiters signal the releaser so that there is no over-pumping
	sem_t waitq;				// The semaphore on which the waiters wait
  public:
	ReductionBarrier(int size);
	
	// Unlike Profe's regular barrier, this barrier takes arguments in the wait function. The first argument
	// refers to the partial result of a reduction computed by the PPU controller thread that made the call.
	// The second argument refers to the LPU property for holding the final reduction result. The wait 
	// function does not process these arguments itself. They are added here so that the function can forward 
	// the arguments to the three plug-point functions 
	void wait(reduction::result *localPartialResult, void *target);
  protected:
	// --------------------------------------------------------------------------------- plug point functions
	
	// This function is invoked when the fist PPU controller thread enters into barrier wait. This can be
	// used to do any initialization needed for the reduction primitives that will use the barrier.
	virtual void initFunction(void *target) {}

	// This function is invoked once for each PPU controller's entrance to the barrier wait. This is intended
	// to be used for accumulating local partial results to some internal data structure within reduction
	// primitives.
	virtual void entryFunction(reduction::result *localPartialResult) {}

	// This function is invoked after all local PPU controller participants of the barrier entered it and the
	// barrier is about to release them. This function can be extended in the subclass to do the communication 
	// for cross-segment reduction.
	virtual void releaseFunction() {}
};

#endif
