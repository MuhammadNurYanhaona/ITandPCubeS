#ifndef SEGMENT_MPI_GROUP_H_
#define SEGMENT_MPI_GROUP_H_

#include <vector>
#include <mpi.h>
#include <iostream>

class SegmentGroup {
private:
	std::vector<int> segments;
	std::vector<int> segmentRanks;
	MPI_Comm mpiCommunicator;
public:
	SegmentGroup(std::vector<int> segments);
	void setupCommunicator();
	MPI_Comm getCommunicator() { return mpiCommunicator; }
	int getRank(int segmentId);
	void describe(std::ostream &stream);
};

#endif
