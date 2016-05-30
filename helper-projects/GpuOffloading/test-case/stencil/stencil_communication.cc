#include "stencil_communication.h"
#include "stencil_structure.h"
#include "../../runtime/structure.h"
#include "../../utils/list.h"

#include <mpi.h>
#include <cstdlib>

//-------------------------------------------------------- Stencil Communicator ---------------------------------------------------------/

StencilComm::StencilComm(int padding, int lpuCount, List<stencil::PlatePart*> *partList) {
	this->padding = padding;
	this->lpuCount = lpuCount;
	this->partList = partList;
	active = (partList->NumElements() > 0);
	if (active > 0) {
		localParts.min = partList->Nth(0)->partId->Nth(0)[0];
		int localPartsCount = partList->NumElements();
		localParts.max = localParts.min + localPartsCount - 1;
	}
}

void StencilComm::synchronizeDataParts() {
	if (active) {
		performLocalExchange(partList);
		performRemoteExchange(partList);
	}
}

void StencilComm::performLocalExchange(List<stencil::PlatePart*> *listOfParts) {

	for (int i = localParts.min + 1; i <= localParts.max; i++) {

		int partIndex = i - localParts.min;
		stencil::PlatePart *dataPart = listOfParts->Nth(partIndex);		
		stencil::PlatePart *prevPart = listOfParts->Nth(partIndex - 1);
		int boundaryElements = dataPart->storageDims[1].getLength() * padding;
		
		// receive from the previous
		int sendIndexInPrev = prevPart->storageDims[0].getLength() 
				* prevPart->storageDims[1].getLength() 
				- 2 * boundaryElements;
		double *dataDst = dataPart->data;
		double *dataSrc = prevPart->data + sendIndexInPrev;
		memcpy(dataDst, dataSrc, boundaryElements * sizeof(double));

		// send to the previous
		int recvIndexInPrev = prevPart->storageDims[0].getLength()
                                * prevPart->storageDims[1].getLength() - boundaryElements;
		dataDst = prevPart->data + recvIndexInPrev;
		dataSrc = dataPart->data + boundaryElements;
		memcpy(dataDst, dataSrc, boundaryElements * sizeof(double));
	}
}

void StencilComm::performRemoteExchange(List<stencil::PlatePart*> *listOfParts) {
	
	if (listOfParts->NumElements() == lpuCount) return;

	int segmentId, segmentCount;
	MPI_Comm_rank(MPI_COMM_WORLD, &segmentId);
	MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

	int currentRequest = 0;
	MPI_Request requests[4];

	// send-receive from the upper boundary
	if (localParts.min != 0) {
		stencil::PlatePart *part = listOfParts->Nth(0);
		double *data = part->data;
		int boundaryElements = part->storageDims[1].getLength() * padding;
		double *sendPtr = data + boundaryElements;
		int status = MPI_Isend(sendPtr, boundaryElements, MPI_DOUBLE,
                                        segmentId - 1, 0, MPI_COMM_WORLD, &requests[currentRequest]);
		if (status != MPI_SUCCESS) {
			std::cout << "send failed\n";
			std::exit(EXIT_FAILURE);
		}
		currentRequest++;

		double *recvPtr = data;
		status = MPI_Irecv(recvPtr, boundaryElements, MPI_DOUBLE, 
				segmentId - 1, 0, MPI_COMM_WORLD, &requests[currentRequest]);
		if (status != MPI_SUCCESS) {
			std::cout << "receive failed\n";
			std::exit(EXIT_FAILURE);
		}
		currentRequest++;
	}

	// send-receve from the lower boundary
	if (localParts.max != lpuCount - 1) {
		stencil::PlatePart *part = listOfParts->Nth(listOfParts->NumElements() - 1);
		double *data = part->data;
		int boundaryElements = part->storageDims[1].getLength() * padding;
		int sendIndex = part->storageDims[0].getLength() 
				* part->storageDims[1].getLength() - 2 * boundaryElements;
		double *sendPtr = data + sendIndex;
		int status = MPI_Isend(sendPtr, boundaryElements, MPI_DOUBLE,
                                        segmentId + 1, 0, MPI_COMM_WORLD, &requests[currentRequest]);
		if (status != MPI_SUCCESS) {
			std::cout << "send failed\n";
			std::exit(EXIT_FAILURE);
		}
		currentRequest++;

		double *recvPtr = sendPtr + boundaryElements;
		status = MPI_Irecv(recvPtr, boundaryElements, MPI_DOUBLE, 
				segmentId + 1, 0, MPI_COMM_WORLD, &requests[currentRequest]);
		if (status != MPI_SUCCESS) {
			std::cout << "receive failed\n";
			std::exit(EXIT_FAILURE);
		}
		currentRequest++;
	}

	// wait for send and receive to complete
	int status = MPI_Waitall(currentRequest, requests, MPI_STATUSES_IGNORE);
        if (status != MPI_SUCCESS) {
                std::cout << "some of the MPI requests failed\n";
                std::exit(EXIT_FAILURE);
        }
}

//------------------------------------------------- Stencil Communicator with Verifier --------------------------------------------------/

StencilCommWithVerifier::StencilCommWithVerifier(int padding, int lpuCount, 
                List<stencil::PlatePart*> *partList, 
                List<stencil::PlatePart*> *duplicatePartList) 
		: StencilComm(padding, lpuCount, partList) {
	this->duplicatePartList = duplicatePartList;
}

void StencilCommWithVerifier::synchronizeDataParts() {
	StencilComm::synchronizeDataParts();
	if (active) {
		performLocalExchange(duplicatePartList);
		performRemoteExchange(duplicatePartList);
	}
}
