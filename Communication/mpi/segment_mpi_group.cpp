#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include "segment_mpi_group.h"

using namespace std;

SegmentGroup::SegmentGroup(vector<int> segments) {
	this->segments = vector<int>(segments);
	this->segmentRanks = vector<int>(segments);
	mpiCommunicator = MPI_COMM_WORLD;
}

void SegmentGroup::setupCommunicator() {

	int segmentRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &segmentRank);

	MPI_Group orig_group, new_group;
	MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

	int participants = segments.size();
	int *originalRanks = new int[participants];
	for (unsigned int i = 0; i < segments.size(); i++) {
		originalRanks[i] = segments.at(i);
	}

	MPI_Group_incl(orig_group, participants, originalRanks, &new_group);
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &mpiCommunicator);

	int groupRank;
	MPI_Comm_rank(mpiCommunicator, &groupRank);

	int participantRanks[participants];
    int status = MPI_Allgather(&segmentRank, 1, MPI_INT,
                    participantRanks, 1, MPI_INT, mpiCommunicator);

    if (status == MPI_ERR_COMM) {
            cout << "invalid communicator\n";
            exit(EXIT_FAILURE);
    } else if (status == MPI_ERR_COUNT) {
            cout << "invalid data element count\n";
            exit(EXIT_FAILURE);
    } else if (status == MPI_ERR_BUFFER) {
            cout << "invalid buffer\n";
            exit(EXIT_FAILURE);
    }

	for (int rank = 0; rank < participants; rank++) {
		int segmentTag = participantRanks[rank];
		for (int i = 0; i < participants; i++) {
			if (segments.at(i) == segmentTag) {
				segmentRanks[i] = rank;
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
