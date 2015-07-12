#include "environment_base.h"
#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "stream.h"

#include <mpi.h>

void EnvironmentBase::getReadyForOutput(int segmentId, int segmentCount, MPI_Comm communicator) {
	
	// first check if there is any output binding specified for the task
	Iterator<const char*> iterator = outputBindings->GetIterator();
	bool hasBinding = (iterator.GetNextValue() != NULL);

	// if the current segment is the first segment then it can proceed immediately; otherwise, 
	// it has to wait for the previous segment to finish writing
	if (segmentId != 0 && hasBinding) {
		int predecessorDone = 0;
		MPI_Recv(&predecessorDone, 1, MPI_INT, segmentId - 1, 
				MPI_ANY_TAG, communicator, MPI_STATUS_IGNORE);
	} 
}

void EnvironmentBase::signalOutputCompletion(int segmentId, int segmentCount, MPI_Comm communicator) {
	
	// first check if there is any output binding specified for the task
	Iterator<const char*> iterator = outputBindings->GetIterator();
	bool hasBinding = (iterator.GetNextValue() != NULL);

	// if the current segment is not the last one, it has to signal the next segment
	if (segmentId != segmentCount - 1 && hasBinding) {
		int writingDone = 1;
		MPI_Send(&writingDone, 1, MPI_INT, segmentId + 1, MPI_ANY_TAG, communicator);
	} 
}
