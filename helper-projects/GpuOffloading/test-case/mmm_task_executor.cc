#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "mmm_structure.h"
#include "../runtime/structure.h"
#include "../utils/partition.h"
#include "../gpu-execution/mm-multiply/mmm_gpu_execution.h"
#include "../gpu-utils/gpu_constant.h"

using namespace std;

Dimension A_MD[2], B_MD[2], C_MD[2];

int *getLpuCount(Dimension dim1, Dimension dim2, int blockSize);
void generateParts(Range myLpuRange, 
		IdGenerator *idGenerator, 
		MatrixPartGenerator *partGenerator, 
		MatrixPartMap *partMap);

void computeCpuMMM(MMMLpu *lpu);

int main(int argc, char *argv[]) {
	
	MPI_Init(&argc, &argv);

        int segmentId;
        int segmentCount;
        MPI_Comm_rank(MPI_COMM_WORLD, &segmentId);
        MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

	int matrixLength, blockSize;
	matrixLength = (argc > 1) ? atoi(argv[1]) : 1000;
	blockSize = (argc > 2) ? atoi(argv[2]) : 32;
	int batchSize = (argc > 3) ?  atoi(argv[3]) : BLOCK_COUNT * 10;

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

	// create duplicates of the data parts to do the same computation in the CPU then compare the results
	MatrixPartMap *duplicatePartMap = partMap->duplicate();
	
	// initialize GPU code executor
	long memLimit = 3 * 1000 * 1000 * 1024l;
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
		
		// at the same time do a CPU execution of the same LPU on duplicate data for verification purpose
		getNextLpu(lpuId, lpu, idGenerator, duplicatePartMap);
		computeCpuMMM(lpu);
	}
	// this is needed to run the last, if exists, partially completed batch that has not run in the GPU 
	gpuExecutor->forceExecution();

	logFile << "cleaning up the GPU state\n";
	logFile.flush();
	// cleanup the GPU code execution context
	gpuExecutor->cleanup();

	// compare CPU and GPU computations' data parts
	logFile << "comparing the result of CPU computation with that of the GPU\n";
	logFile.flush();
	partMap->matchParts(duplicatePartMap, logFile);
	
	logFile << "finished task execution\n";
	logFile.flush();
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

void computeCpuMMM(MMMLpu *lpu) {

	double *a, *b, *c;
	a = lpu->a;
	b = lpu->b;
	c = lpu->c;
	
	Dimension aStoreDims[2], bStoreDims[2], cStoreDims[2];
	aStoreDims[0] = lpu->aPartDims[0].storage;
	aStoreDims[1] = lpu->aPartDims[1].storage;
	bStoreDims[0] = lpu->bPartDims[0].storage;
	bStoreDims[1] = lpu->bPartDims[1].storage;
	cStoreDims[0] = lpu->cPartDims[0].storage;
	cStoreDims[1] = lpu->cPartDims[1].storage;

	Dimension aPartDims[2], bPartDims[2], cPartDims[2];
	aPartDims[0] = lpu->aPartDims[0].partition;
	aPartDims[1] = lpu->aPartDims[1].partition;
	bPartDims[0] = lpu->bPartDims[0].partition;
	bPartDims[1] = lpu->bPartDims[1].partition;
	cPartDims[0] = lpu->cPartDims[0].partition;
	cPartDims[1] = lpu->cPartDims[1].partition;

	for (int i = aPartDims[0].range.min; i <= aPartDims[0].range.max; i++) {
		int a_i = i - aStoreDims[0].range.min;
		int c_i = i - cStoreDims[0].range.min;
		for (int j = bPartDims[1].range.min; j <= bPartDims[1].range.max; j++) {
			int b_j = j - bStoreDims[1].range.min;
			int c_j = j - cStoreDims[1].range.min;
			for (int k = aPartDims[1].range.min; k <= aPartDims[1].range.max; k++) {
				int a_k = k - aStoreDims[1].range.min;
				int b_k = k - bStoreDims[0].range.min;
				int a_index = a_i * aStoreDims[1].getLength() + a_k;
				int b_index = b_k * bStoreDims[1].getLength() + b_j;
				int c_index = c_i * cStoreDims[1].getLength() + c_j;
				c[c_index] += a[a_index] * b[b_index];
			}
		}
	}
}
