#ifndef _H_reduction
#define _H_reduction

#include "../utils/common_constant.h"

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <fstream>

// forward declaration of the class that creates and holds MPI communicators
class SegmentGroup;

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

/* This is another extension of Profe's barrier class. It is designed for implementing segment-local and cross-
   segment reductions. It provides three plug points to insert custom, context dependent logic inside the synchro-
   nization process.	    
*/
class ReductionBarrier {
  private:
	int _size;				// How many threads need call wait before releasing all threads 
	int _count;				// Current count of waiting threads
	sem_t mutex;  				// The mutex
	sem_t throttle;				// Waiters signal the releaser so that there is no over-pumping
	sem_t waitq;				// The semaphore on which the waiters wait
  public:
	ReductionBarrier(int size);
	
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
	virtual void initFunction(reduction::Result *localPartialResult, void *target) {}

	// This function is invoked once for each PPU controller's entrance to the barrier wait. This is intended
	// to be used for accumulating local partial results to some internal data structure within reduction
	// primitives.
	virtual void entryFunction(reduction::Result *localPartialResult) {}

	// This function is invoked after all local PPU controller participants of the barrier entered it and the
	// barrier is about to release them. This function can be extended in the subclass to do the communication 
	// for cross-segment reduction.
	virtual void releaseFunction() {}
};

/* This extension of the Reduction-Barrier embodies the logic for doing reduction of partial results computed by 
 * PPU controllers local to the current segment. If the final reduction is localized to individual segments then
 * this class should be extended. Otherwise, the following class should be extended.  
 */
class ReductionPrimitive : public ReductionBarrier {
  protected:
	void *target;
	int elementSize;
	reduction::Result *intermediateResult;
	ReductionOperator op;
	std::ofstream *logFile;
  public:
	ReductionPrimitive(int elementSize, ReductionOperator op, int localParticipants);
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }

	// Different reduction function requires different initial values for the partial result variable -- the
	// result variable cannot be just set to all zeros. So subclasses should provide proper implementations.
	virtual void resetPartialResult(reduction::Result *resultVar) = 0;
  protected:
	void initFunction(reduction::Result *localPartialResult, void *target);
	void entryFunction(reduction::Result *localPartialResult) {
		updateIntermediateResult(localPartialResult);
	}
	virtual void releaseFunction();

	// This is the only function a subclass has to implement. This specifies how the result computed by the 
	// calling PPU controller is applied to the partial result so far computed by incrementally applying 
	// results from earlier PPU controllers. Note that the result of first PPU controller is handled by the
	// initFunction(). This function is needed for subsequent PPU controllers.
	virtual void updateIntermediateResult(reduction::Result *localPartialResult) = 0;	 
};

/* This extension of the Reduction-Primitive is needed for cross-segment reduction operation. The reduction of 
 * the partial results computed in individual segments are done at the end using MPI communication(s). A subclass
 * should specifies how the MPI communication(s) is(are) done.
 */
class MpiReductionPrimitive : public ReductionPrimitive {
  protected:
	SegmentGroup *segmentGroup;

	// these two buffers are used for sending local results and receiving final results respectively. Care
	// should be taken so that the data and/or index of reduction are accessed correctly from these buffers 
	// in the subclass. The subclass implementer should investigate the implementation of releaseFunction()
	// function to avoid mistakes.
	char *sendBuffer;
	char *receiveBuffer;
  public:
	MpiReductionPrimitive(int elementSize,
			ReductionOperator op, 
			int localParticipants, 
			SegmentGroup *segmentGroup);
  protected:
	void releaseFunction();

	// This is the function that a subclass has to implement with addition to the updateIntermediateResult()
	// function of the superclass. This function specifies how MPI communication is done at the end to carry
	// out the final step of the cross-segment reduction.
	virtual void performCrossSegmentReduction() = 0;
};

#endif
