#include "environment_base.h"
#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "stream.h"

#include <mpi.h>
#include <fstream>

void EnvironmentBase::getReadyForOutput(int segmentId, int segmentCount, MPI_Comm communicator, std::ofstream &logFile) {
	
	// first check if there is any output binding specified for the task
	Iterator<const char*> iterator = outputBindings->GetIterator();
	bool hasBinding = (iterator.GetNextValue() != NULL);

	// if the current segment is the first segment then it can proceed immediately; otherwise, 
	// it has to wait for the previous segment to finish writing
	if (segmentId != 0 && hasBinding) {
		logFile << "\t\twaiting for segment #" << segmentId - 1 << " to signal its output completion\n";
		logFile.flush();
		int predecessorDone = 0;
		MPI_Status status;
		MPI_Recv(&predecessorDone, 1, MPI_INT, segmentId - 1, 0, communicator, &status);
	} 
}

void EnvironmentBase::signalOutputCompletion(int segmentId, int segmentCount, MPI_Comm communicator, std::ofstream &logFile) {
	
	// first check if there is any output binding specified for the task
	Iterator<const char*> iterator = outputBindings->GetIterator();
	bool hasBinding = (iterator.GetNextValue() != NULL);

	// if the current segment is not the last one, it has to signal the next segment
	if (segmentId != segmentCount - 1 && hasBinding) {
		logFile << "\t\tsignaling segment #" << segmentId + 1 << " about own output completion\n";
		MPI_Request sendRequest;
		int writingDone = 1;
		MPI_Isend(&writingDone, 1, MPI_INT, segmentId + 1, 0, communicator, &sendRequest);
	} 
}
