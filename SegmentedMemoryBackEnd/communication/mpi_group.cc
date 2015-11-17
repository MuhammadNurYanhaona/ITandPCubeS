#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "mpi_group.h"

using namespace std;

SegmentGroup::SegmentGroup(vector<int> segments) {
        this->segments = vector<int>(segments);
        this->segmentRanks = vector<int>(segments);
        mpiCommunicator = MPI_COMM_WORLD;
}

void SegmentGroup::setupCommunicator(std::ofstream &log) {

        int segmentRank, segmentCount;
        MPI_Comm_rank(MPI_COMM_WORLD, &segmentRank);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);
        
	int participants = segments.size();
	if (participants == segmentCount) return;

	// the ID of the communicator group is the ID of the first segment; this gives individual groups their unique IDs
	int color = segments.at(0);
	int status = MPI_Comm_split(MPI_COMM_WORLD, color, segmentRank, &mpiCommunicator);
	if (status != MPI_SUCCESS) {
		log << "\tcould not create a new communicator for the group\n";
		log.flush();
		exit(EXIT_FAILURE);
	}

        int groupRank;
        MPI_Comm_rank(mpiCommunicator, &groupRank);

	// at setup time, each segment in the current group should retrieve the group ranks of all other segments to be 
	// able to contact them later by their group Ids.
        int participantRanks[participants];
        status = MPI_Allgather(&segmentRank, 1, MPI_INT, participantRanks, 1, MPI_INT, mpiCommunicator);
	if (status != MPI_SUCCESS) {
		log << "\tcould not gather rank information of segments in the new communicator\n";
		log.flush();
		exit(EXIT_FAILURE);
	}

        for (int rank = 0; rank < participants; rank++) {
                int segmentTag = participantRanks[rank];
                for (int i = 0; i < participants; i++) {
                        if (segments.at(i) == segmentTag) {
                                segmentRanks.at(i) = rank;
                                break;
                        }
                }
        }
}

int SegmentGroup::getRank(int segmentId) {
        for (unsigned int i = 0; i < segments.size(); i++) {
                if (segments.at(i) == segmentId) {
                        return segmentRanks.at(i);
                }
        }
        return -1;
}

void SegmentGroup::describe(std::ostream &stream) {
        stream << "Segment Group:\n";
        for (unsigned int i = 0; i < segments.size(); i++) {
                stream << '\t' << "Segment Id: " << segments.at(i) << ", ";
                stream << "Group Rank:" << segmentRanks.at(i) << "\n";
        }
}

void SegmentGroup::excludeSegmentFromGroupSetup(int segmentId, std::ofstream &log) {
	MPI_Comm nullComm;
	int status = MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, segmentId, &nullComm);
	if (status != MPI_SUCCESS) {
		log << '\t' << segmentId << ": could not exclude myself from the restricted communicator\n";
		log.flush();
		exit(EXIT_FAILURE);
	}
}
