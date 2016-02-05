#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include "../mpi/segment_mpi_group.h"

using namespace std;

int mainMPICT(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int rank;
	int segmentCount;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

	int groupNo = rank % 3;
	vector<int> segmentsInGroup;
	for (int i = 0; i < segmentCount; i++) {
		if (i % 3 == groupNo) segmentsInGroup.push_back(i);
	}
	SegmentGroup *group = new SegmentGroup(segmentsInGroup);
	group->setupCommunicator();

	ostringstream message;
	message << "Process #" << rank << "'s ";
	group->describe(message);
	cout << message.str();
	cout.flush();

	MPI_Finalize();

	return 0;
}
