#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>

#include "stencil_structure.h"
#include "stencil_communication.h"
#include "../../gpu-execution/stencil/stencil_gpu_execution.h"
#include "../../runtime/structure.h"
#include "../../utils/partition.h"
#include "../../gpu-utils/gpu_constant.h"

using namespace std;

Dimension Plate_MD[2];

void generateParts(Range myLpuRange,
                stencil::IdGenerator *idGenerator,
                stencil::PlatePartGenerator *partGenerator,
                stencil::PlatePartMap *partMap);

void computeCpuStencil(stencil::StencilLpu *lpu, stencil::Partition partition);

int mainStencil(int argc, char *argv[]) {

        MPI_Init(&argc, &argv);
	        
	// start the execution timer
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long startTime = tv.tv_sec * 1000000 + tv.tv_usec;

        int segmentId;
        int segmentCount;
        MPI_Comm_rank(MPI_COMM_WORLD, &segmentId);
        MPI_Comm_size(MPI_COMM_WORLD, &segmentCount);

        int plateDim = (argc > 1) ? atoi(argv[1]) : 1000;
        int blockCount = (argc > 2) ? atoi(argv[2]) : segmentCount * BLOCK_COUNT;
        int padding1 = (argc > 3) ? atoi(argv[3]) : 10;
        int blockSize = (argc > 4) ? atoi(argv[4]) : 32;
        int padding2 = (argc > 5) ? atoi(argv[5]) : 1;
	int iterations = (argc > 6) ? atoi(argv[6]) : 1000;
        int batchSize = (argc > 7) ?  atoi(argv[7]) : BLOCK_COUNT * 10;
        bool verifyCorrectness = (argc > 8) ? (atoi(argv[8]) == 1) : false;

        Plate_MD[0].setLength(plateDim);
        Plate_MD[1] = Plate_MD[0];

        // create a log file
        ostringstream fileName;
        fileName << "process_" << segmentId << ".log";
        ofstream logFile;
        logFile.open(fileName.str().c_str(), ofstream::out | ofstream::app);
        if (!logFile.is_open()) {
                cout << "Could not open log file for Process-" << segmentId << "\n";
        }

	// initialize useful scalar data structures needed for both host and GPU computations
        stencil::Partition partition;
        partition.blockCount = blockCount;
	partition.blockSize = blockSize;
	partition.padding1 = padding1;
	partition.padding2 = padding2;
        stencil::ArrayMetadata arrayMetadata;
        arrayMetadata.plateDims[0] = Plate_MD[0];
        arrayMetadata.plateDims[1] = Plate_MD[1];
        stencil::TaskGlobals *taskGlobals = new stencil::TaskGlobals();
	taskGlobals->iterations = iterations;
        stencil::ThreadLocals *threadLocals = new stencil::ThreadLocals();
	threadLocals->currIteration = 0;

	// determine the ranges of LPUs the current process is responsible for
	int spaceALpuCount = block_count_partitionCount(Plate_MD[0], blockCount);
	int perProcessLpus = (spaceALpuCount + segmentCount - 1) / segmentCount;
        Range myLpuRange;
        myLpuRange.min = segmentId * perProcessLpus;
        myLpuRange.max = myLpuRange.min + perProcessLpus - 1;
        if (myLpuRange.max >= spaceALpuCount) myLpuRange.max = spaceALpuCount - 1;

	// generate and save the data parts
        stencil::IdGenerator *idGenerator = new stencil::IdGenerator(spaceALpuCount);
        stencil::PlatePartGenerator *partGenerator = new stencil::PlatePartGenerator(spaceALpuCount,
                        padding1, &Plate_MD[0]);
        stencil::PlatePartMap *partMap = new stencil::PlatePartMap();
        generateParts(myLpuRange, idGenerator, partGenerator, partMap);

	// create duplicates of the data parts to do the same computation in the CPU then compare the results
        stencil::PlatePartMap *duplicatePartMap = NULL;
        if (verifyCorrectness) {
                duplicatePartMap = partMap->duplicate();
        }

	// instantiate a communicator to synchronize data parts of different processes
	StencilComm *communicator = NULL;
	if (verifyCorrectness) {
		communicator = new StencilCommWithVerifier(padding1, spaceALpuCount,
                		partMap->getPartList(), 
				duplicatePartMap->getPartList());
	} else {
		communicator = new StencilComm(padding1, spaceALpuCount, partMap->getPartList());
	}

	// initialize GPU code executor
        long memLimit = 3 * 1000 * 1000 * 1024l;
        StencilLpuBatchController *lpuBatchController
                        = new StencilLpuBatchController(batchSize, memLimit);
        lpuBatchController->setLogFile(&logFile);
        StencilGpuCodeExecutor *gpuExecutor = new StencilGpuCodeExecutor(lpuBatchController,
                        partition, arrayMetadata, taskGlobals, threadLocals);
        gpuExecutor->setLpuCount(&spaceALpuCount);
        gpuExecutor->setLogFile(&logFile);
        gpuExecutor->initialize();

	// execute the stencil computation flow
	stencil::StencilLpu *lpu = new stencil::StencilLpu();
	int currIter = 0;
	while (currIter < iterations) {

		// offload Space A LPUs to the GPU offloader
		for (int lpuId = myLpuRange.min; lpuId <= myLpuRange.max; lpuId++) {
			stencil::getNextLpu(lpuId, lpu, idGenerator, partMap);
                	gpuExecutor->submitNextLpu(lpu);
			
			// perform duplicate computation in the host CPU in the verification mode
			if (verifyCorrectness) {
				stencil::getNextLpu(lpuId, lpu, idGenerator, duplicatePartMap);
                        	computeCpuStencil(lpu, partition);	
			}
		}

		// this is needed to run the last, if exists, partially completed batch that has not run in the GPU 
        	gpuExecutor->forceExecution();

		// do MPI communications to synchronize Space A data parts among different processes
		communicator->synchronizeDataParts();

		currIter++;
	}

	// compare CPU and GPU computations' data parts
        if (verifyCorrectness) {
                logFile << "comparing the result of CPU computation with that of the GPU\n";
                logFile.flush();
                partMap->matchParts(duplicatePartMap, logFile);
        }

        // record program's running time
        gettimeofday(&tv, NULL);
        long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
        double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
        logFile << "Total time taken by the program: " << timeTaken << "\n";

	logFile << "finished task execution\n";
        logFile.flush();
        MPI_Finalize();
        return 1;
}

void generateParts(Range myLpuRange,
                stencil::IdGenerator *idGenerator,
                stencil::PlatePartGenerator *partGenerator,
                stencil::PlatePartMap *partMap) {
	
	for (int lpuId = myLpuRange.min; lpuId <= myLpuRange.max; lpuId++) {
                List<int*> *partId = idGenerator->getPartId(lpuId);
        	partMap->addPart(partGenerator->generatePart(partId));
        }
}

void computeCpuStencil(stencil::StencilLpu *lpu, stencil::Partition partition) {
	
	int firstRow = lpu->platePartDims[0].partition.range.min;
	int lastRow = lpu->platePartDims[0].partition.range.max;
	int firstCol = lpu->platePartDims[1].partition.range.min;
	int lastCol = lpu->platePartDims[1].partition.range.max;
	int colCount = lpu->platePartDims[1].partition.getLength();
	

	for (int i = 0; i < partition.padding1; i++) {
		
		lpu->partReference->advanceEpoch();
		
		double *data = lpu->partReference->getData(0);
		double *data_lag_1 = lpu->partReference->getData(1);

		for (int r = firstRow + 1; r < lastRow; r++) {	
			for (int c = firstCol + 1; c < lastCol; c++) {
				data[(r - firstRow) * colCount + (c - firstCol)] = 0.25 * (
						  data_lag_1[(r + 1 - firstRow) * colCount + (c - firstCol)]
						+ data_lag_1[(r - 1 - firstRow) * colCount + (c - firstCol)]
						+ data_lag_1[(r - firstRow) * colCount + (c + 1 - firstCol)]
						+ data_lag_1[(r - firstRow) * colCount + (c - 1 - firstCol)]);
			}
		}
	}
}
