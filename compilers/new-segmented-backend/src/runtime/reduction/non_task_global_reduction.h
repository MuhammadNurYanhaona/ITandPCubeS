#ifndef _H_non_task_global_reduction
#define _H_non_task_global_reduction

#include "reduction_barrier.h"
#include "../../../../common-libs/domain-obj/constant.h"

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>
#include <fstream>

// forward declaration of the class that creates and holds MPI communicators
class SegmentGroup;

/* This extension of the Reduction-Barrier embodies the logic for doing reduction of partial results computed by 
 * PPU controllers local to the current segment. If the final reduction is localized to individual segments then
 * this class should be extended. Otherwise, the following class should be extended.  
 */
class NonTaskGlobalReductionPrimitive : public NonTaskGlobalReductionBarrier {
  protected:
	int elementSize;
	ReductionOperator op;
	std::ofstream *logFile;
  public:
	NonTaskGlobalReductionPrimitive(int elementSize, ReductionOperator op, int localParticipants);
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }

	// Different reduction function requires different initial values for the partial result variable -- the
	// result variable cannot be just set to all zeros. So subclasses should provide proper implementations.
	virtual void resetPartialResult(reduction::Result *resultVar) = 0;
  protected:
	void entryFunction(reduction::Result *localPartialResult) {
		updateIntermediateResult(localPartialResult);
	}
	void updateLocalTarget(reduction::Result *finalResult, void *currLocalTarget);

	// This is the second function a subclass has to implement. This specifies how the result computed by the 
	// calling PPU controller is applied to the partial result so far computed by incrementally applying 
	// results from earlier PPU controllers. Note that the result of first PPU controller is handled by the
	// superclass's initFunction(). This function is needed for subsequent PPU controllers.
	virtual void updateIntermediateResult(reduction::Result *localPartialResult) = 0;	 
};

/* This extension of the Reduction-Primitive is needed for cross-segment reduction operation. The reduction of 
 * the partial results computed in individual segments are done at the end using MPI communication(s). A subclass
 * should specifies how the MPI communication(s) is(are) done.
 */
class NonTaskGlobalMpiReductionPrimitive : public NonTaskGlobalReductionPrimitive {
  protected:
	SegmentGroup *segmentGroup;

	// these two buffers are used for sending local results and receiving final results respectively. Care
	// should be taken so that the data and/or index of reduction are accessed correctly from these buffers 
	// in the subclass. The subclass implementer should investigate the implementation of releaseFunction()
	// function to avoid mistakes.
	char *sendBuffer;
	char *receiveBuffer;
  public:
	NonTaskGlobalMpiReductionPrimitive(int elementSize,
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
