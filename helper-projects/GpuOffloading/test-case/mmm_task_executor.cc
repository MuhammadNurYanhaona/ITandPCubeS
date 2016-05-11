#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "mmm_structure.h"
#include "../runtime/structure.h"
#include "../utils/partition.h"
#include "../gpu-execution/mm-multiply/mmm_gpu_execution.h"

using namespace std;

Dimension A_MD[2], B_MD[2], C_MD[2];

int *getLpuCount(Dimension dim1, Dimension dim2, int blockSize);
void generateParts(Range myLpuRange, 
		IdGenerator *idGenerator, 
		MatrixPartGenerator *partGenerator, 
		MatrixPartMap *partMap);

int main(int argc, char *argv[]) {
	
	MPI_Init(&argc, &argv);

        int segmentId;
        int segmentCount;
        MPI_Comm_rank(MPI_COMM_WORLD, &segmentId);
        MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

	int matrixLength, blockSize;
	matrixLength = (argc > 1) ? atoi(argv[1]) : 1000;
	blockSize = (argc > 2) ? atoi(argv[2]) : 32;

	A_MD[0].setLength(matrixLength);
	C_MD[1] = C_MD[0] = B_MD[1] = B_MD[0] = A_MD[1] = A_MD[0];

	// create a log file
        std::ostringstream fileName;
        fileName << "process_" << segmentId << ".log";
	std::ofstream logFile;
        logFile.open(fileName.str().c_str(), std::ofstream::out | std::ofstream::app);
        if (!logFile.is_open()) {
                std::cout << "Could not open log file for Process-" << segmentId << "\n";
        }

	// initialize useful scalar data structures needed for both host and GPU computations
	mmm::Partition partition;
	partition.blockSize = blockSize;
	mmm::ArrayMetadata arrayMetadata;
	arrayMetadata.aDims[0] = A_MD[0];
	arrayMetadata.aDims[1] = A_MD[1];
	arrayMetadata.bDims[0] = B_MD[0];
	arrayMetadata.bDims[1] = B_MD[1];
	arrayMetadata.cDims[0] = C_MD[0];
	arrayMetadata.cDims[1] = C_MD[1];
	mmm::TaskGlobals *taskGlobals = new mmm::TaskGlobals();
	mmm::ThreadLocals *threadLocals = new mmm::ThreadLocals();
	
	// determine the ranges of LPUs the current process is responsible for
	int *lpuCount = getLpuCount(C_MD[0], C_MD[1], blockSize);
	int linearLpuCount = lpuCount[0] * lpuCount[1];
	int perProcessLpus = (linearLpuCount + segmentCount - 1) / segmentCount;
	Range myLpuRange;
	myLpuRange.min = segmentId * perProcessLpus;
	myLpuRange.max = myLpuRange.min + perProcessLpus - 1;
	if (myLpuRange.max >= linearLpuCount) myLpuRange.max = linearLpuCount - 1;

	// generate and save the data parts
	IdGenerator *idGenerator = new IdGenerator(lpuCount);
	MatrixPartGenerator *partGenerator = new MatrixPartGenerator(lpuCount, 
			blockSize, &A_MD[0], &B_MD[0], &C_MD[0]);
	MatrixPartMap *partMap = new MatrixPartMap();
	generateParts(myLpuRange, idGenerator, partGenerator, partMap); 
	
	// initialize GPU code executor
	long memLimit = 3 * 1000 * 1000 * 1024l;
	int batchSize = 100;
	MMMLpuBatchController *lpuBatchController 
			= new MMMLpuBatchController(batchSize, memLimit);
	lpuBatchController->setLogFile(&logFile);	
	MMMGpuCodeExecutor *gpuExecutor = new MMMGpuCodeExecutor(lpuBatchController, 
			partition, arrayMetadata, taskGlobals, threadLocals);
	gpuExecutor->setLpuCount(lpuCount);
	gpuExecutor->setLogFile(&logFile);
	gpuExecutor->initialize();

	// offload LPUs to the gpu code executor
	MMMLpu *lpu = new MMMLpu();
	for (int lpuId = myLpuRange.min; lpuId <= myLpuRange.max; lpuId++) {
		getNextLpu(lpuId, lpu, idGenerator, partMap);
		gpuExecutor->submitNextLpu(lpu);
	}
	// this is needed to run the last, if exists, partially completed batch that has not run in the GPU 
	gpuExecutor->forceExecution();

	// cleanup the GPU code execution context
	gpuExecutor->cleanup();
	
	MPI_Finalize();
	return 1;
}

void generateParts(Range myLpuRange,
                IdGenerator *idGenerator,
                MatrixPartGenerator *partGenerator,
                MatrixPartMap *partMap) {

	for (int lpuId = myLpuRange.min; lpuId <= myLpuRange.max; lpuId++) {

		List<int*> *aPartId = idGenerator->getAPartId(lpuId);
		if (!partMap->aPartExists(aPartId)) {
			partMap->addAPart(partGenerator->generateAPart(aPartId));
		} else {
			int *aId = aPartId->Nth(0);
			delete[] aId;
			delete aPartId;
		}

		List<int*> *bPartId = idGenerator->getBPartId(lpuId);
		if (!partMap->bPartExists(bPartId)) {
			partMap->addBPart(partGenerator->generateBPart(bPartId));
		} else {
			int *bId = bPartId->Nth(0);
			delete[] bId;
			delete bPartId;
		}

		List<int*> *cPartId = idGenerator->getCPartId(lpuId);
		if (!partMap->cPartExists(cPartId)) {
			partMap->addCPart(partGenerator->generateCPart(cPartId));
		} else {
			int *cId = cPartId->Nth(0);
			delete[] cId;
			delete cPartId;
		}
	}	
}

int *getLpuCount(Dimension dim1, Dimension dim2, int blockSize) {
	int *count = new int[2];
	count[0] = block_size_partitionCount(dim1, blockSize);
	count[1] = block_size_partitionCount(dim2, blockSize);
	return count; 
}
