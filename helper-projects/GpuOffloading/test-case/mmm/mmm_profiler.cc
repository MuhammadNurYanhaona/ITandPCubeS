#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>

#include "mmm_structure.h"
#include "../../runtime/structure.h"
#include "../../utils/partition.h"
#include "../../gpu-execution/mm-multiply/mmm_gpu_execution.h"
#include "../../gpu-utils/gpu_constant.h"

using namespace std;

int *getLpuCountPr(Dimension dim1, Dimension dim2, int blockSize);
void generatePartsPr(Range myLpuRange, 
		mmm::IdGenerator *idGenerator, 
		mmm::MatrixPartGenerator *partGenerator, 
		mmm::MatrixPartMap *partMap);

int mainMMMProfile(int argc, char *argv[]) {
	
	Dimension A_MD[2], B_MD[2], C_MD[2];

	// start the execution timer
	struct timeval tv;
        gettimeofday(&tv, NULL);
        long startTime = tv.tv_sec * 1000000 + tv.tv_usec;

        int segmentId = 0;
        int segmentCount = 1;

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
	int *lpuCount = getLpuCountPr(C_MD[0], C_MD[1], blockSize);
	int linearLpuCount = lpuCount[0] * lpuCount[1];
	int perProcessLpus = (linearLpuCount + segmentCount - 1) / segmentCount;
	Range myLpuRange;
	myLpuRange.min = segmentId * perProcessLpus;
	myLpuRange.max = myLpuRange.min + perProcessLpus - 1;
	if (myLpuRange.max >= linearLpuCount) myLpuRange.max = linearLpuCount - 1;

	// generate and save the data parts
	mmm::IdGenerator *idGenerator = new mmm::IdGenerator(lpuCount);
	mmm::MatrixPartGenerator *partGenerator = new mmm::MatrixPartGenerator(lpuCount, 
			blockSize, &A_MD[0], &B_MD[0], &C_MD[0]);
	mmm::MatrixPartMap *partMap = new mmm::MatrixPartMap();
	generatePartsPr(myLpuRange, idGenerator, partGenerator, partMap); 

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
	mmm::MMMLpu *lpu = new mmm::MMMLpu();
	for (int lpuId = myLpuRange.min; lpuId <= myLpuRange.max; lpuId++) {
		getNextLpu(lpuId, lpu, idGenerator, partMap);
		gpuExecutor->submitNextLpu(lpu);
	}
	// this is needed to run the last, if exists, partially completed batch that has not run in the GPU 
	gpuExecutor->forceExecution();

	logFile << "cleaning up the GPU state\n";
	logFile.flush();
	// cleanup the GPU code execution context
	gpuExecutor->cleanup();

	// record program's running time
	gettimeofday(&tv, NULL);
        long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
        double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	logFile << "Total time taken by the program: " << timeTaken << "\n";
	
	logFile << "finished task execution\n";
	logFile.flush();
	return 1;
}

void generatePartsPr(Range myLpuRange,
                mmm::IdGenerator *idGenerator,
                mmm::MatrixPartGenerator *partGenerator,
                mmm::MatrixPartMap *partMap) {

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
		partMap->addCPart(partGenerator->generateCPart(cPartId));
	}	
}

int *getLpuCountPr(Dimension dim1, Dimension dim2, int blockSize) {
	int *count = new int[2];
	count[0] = block_size_partitionCount(dim1, blockSize);
	count[1] = block_size_partitionCount(dim2, blockSize);
	return count; 
}
