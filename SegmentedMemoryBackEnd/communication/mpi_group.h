#ifndef _H_mpi_group
#define _H_mpi_group

// this sole class in this header is used to manage an exclusive MPI-communicator per dependency arc for interacting segments

#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

class SegmentGroup {
  private:
        std::vector<int> segments;
        std::vector<int> segmentRanks;
        MPI_Comm mpiCommunicator;
  public:
	// constructor and setup function to be used when the current segment is unaware who else will be interacting with it
	SegmentGroup();
	void discoverGroupAndSetupCommunicator(std::ofstream &log);
        
	// constructor and setup function to be used when the current segment knows the other segments that will be interacting
	// with it
        SegmentGroup(std::vector<int> segments);
	void setupCommunicator(std::ofstream &log);

        MPI_Comm getCommunicator() { return mpiCommunicator; }
        int getRank(int segmentId);
	int getSegment(int rank) { return segments.at(rank); }
        void describe(std::ostream &stream);
	int getParticipantsCount() { return segments.size(); }
	static void excludeSegmentFromGroupSetup(int segmentId, std::ofstream &log);
};

#endif
