#ifndef _H_reduction
#define _H_reduction

#include "../../../../common-libs/utils/list.h"

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <fstream>

namespace reduction {

	typedef union {
		bool boolValue;
		char charValue;	
		int intValue;
		float floatValue;
		double doubleValue;
		long longValue;
	} Data;

	// The result of a reduction can be the data (e.g., max, sum) or the index (e.g., maxEntry, minEntry). 
	// Since the parallel for loop holding a reduction operation may be iterating on multiple indices, the 
	// index should be multidimensional. But the index here is a single integer -- not an array. So index 
	// based reduction on multidimensional array will not work. We will fix this problem after reduction of
	// data is fully functional. 
	class Result {
	  public:   
		reduction::Data data; 
		unsigned int index;
	};
}

/* This is another extension of Profe's barrier class. It is designed for implementing task-global reductions, i.e.,
 * reduction operations that produce a single result for the entire task. The class provides three plug points to 
 * insert custom, context dependent logic inside the synchronization process.	    
 */
class TaskGlobalReductionBarrier {
  private:
	int _size;				// How many threads need call wait before releasing all threads 
	int _count;				// Current count of waiting threads
	sem_t mutex;  				// The mutex
	sem_t throttle;				// Waiters signal the releaser so that there is no over-pumping
	sem_t waitq;				// The semaphore on which the waiters wait
  public:
	TaskGlobalReductionBarrier(int size);
	
	// Unlike Profe's regular barrier, this barrier takes arguments in the wait (a.k.a reduce) function. The 
	// first argument refers to the partial result of a reduction computed by the PPU controller thread that 
	// made the call. The second argument refers to the LPU property for holding the final reduction result. 
	// The wait function does not process these arguments itself. They are added here so that the function can 
	// forward the arguments to the three plug-point functions 
	void reduce(reduction::Result *localPartialResult, void *target);
  protected:
	// --------------------------------------------------------------------------------- plug point functions
	
	// This function is invoked when the fist PPU controller thread enters into barrier wait. This can be
	// used to do any initialization needed for the reduction primitives that will use the barrier.
	virtual void initFunction(reduction::Result *localPartialResult, void *target) = 0;

	// This function is invoked once for each PPU controller's entrance to the barrier wait. This is intended
	// to be used for accumulating local partial results to some internal data structure within reduction
	// primitives.
	virtual void entryFunction(reduction::Result *localPartialResult) = 0;

	// This function is invoked after all local PPU controller participants of the barrier entered it and the
	// barrier is about to release them. This function can be extended in the subclass to do the communication 
	// for cross-segment reduction.
	virtual void releaseFunction() = 0;
};

/* This modification to the generation barrier class is for reduction operations that executes separately in sub-
 * trees of the LPU hierarchy. In other words, there are multiple versions of results. The mode of operations and
 * plug points of this barrier is similar to the one preceeding it. However, it excepts and updates results of
 * of individual PPUs separately to account for the different memory management and access structure for the
 * result of a non-task-global reduction.  
 */
class NonTaskGlobalReductionBarrier {
  private:
	int _size;				// How many threads need call wait before releasing all threads 
	int _count;				// Current count of waiting threads
	sem_t mutex;  				// The mutex
	sem_t throttle;				// Waiters signal the releaser so that there is no over-pumping
	sem_t waitq;				// The semaphore on which the waiters wait

	List<void*> *localTargets;		// The list containing the local target variables of individual
						// PPU controllers; all these variables have to be updated at
						// the end of the reduction operation.
  protected:	
	reduction::Result *intermediateResult;	// A reference to the final result variable reference that will
						// persist for the entire task execution; the property has given
						// this unusual name to match a similar property in task-global
						// reduction; this similarity simplifies code generation for
						// sub-classes  
  public:
	NonTaskGlobalReductionBarrier(int size);

	// The reduce funtion here takes input the local target of the invoker PPU controller along with its
	// computed partial result. In addition, there is a third input for the reference result variable that
	// will hold the outcome for persistence. 
	void reduce(reduction::Result *localPartialResult, 
			void *localTarget, 
			reduction::Result *toBeStoredFinalResult);
	
	// function being invoked when the first PPU controller call the reduce function
	void initFunction(reduction::Result *localPartialResult, reduction::Result *toBeStoredFinalResult);

	// function being invoked when subsequent PPU controllers call the reduce function; this interface is
	// be used for updating the intermediate result
	virtual void entryFunction(reduction::Result *localPartialResult) = 0;

	// This function is invoked after all local PPU controller participants of the barrier entered it and the
	// barrier is about to release them. This function can be extended in the subclass to do the communication 
	// for cross-segment reduction.
	virtual void releaseFunction() {};
  protected:
	// Subclasses should provide an implementation for this function to update the entries in local-targets
	// with the final result of reduction
	virtual void updateLocalTarget(reduction::Result *finalResult, void *currLocalTarget) = 0;

  private:	
	// This function is invoked at the end to update all local targets of individual PPU controllers
	void updateAllLocalTargets();

	// This function wraps the releaseFunction() with additional instructions for cleaning up auxiliary data
	void executeFinalStepOfReduction();	 
};

#endif
