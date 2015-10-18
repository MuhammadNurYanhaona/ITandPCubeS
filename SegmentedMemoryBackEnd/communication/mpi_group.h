#ifndef _H_mpi_group
#define _H_mpi_group

// this sole class in this header is used to manage an exclusive MPI-communicator per dependency arc for interacting segments

#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <vector>

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
	int getParticipantsCount() { return segments.size(); }
};

#endif
