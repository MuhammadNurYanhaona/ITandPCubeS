#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <vector>

//-------------------------------------------------------- Offload Statistics -------------------------------------------------------------/

OffloadStats::OffloadStats() {
	timeSpentStagingIn = 0;
	timeSpentExecution = 0;
	timeSpentStagingOut = 0;
}

void OffloadStats::describe(std::ofstream &logFile) {
	logFile << "Overall time spent staging data into GPU from CPU: ";
	logFile << timeSpentStagingIn << " Seconds\n";
	logFile << "Overall time spent executing kernels for the LPUs: ";
	logFile << timeSpentExecution << " Seconds\n";
	logFile << "Overall time spent staging data out from GPU to CPU: ";
	logFile << timeSpentStagingOut << " Seconds\n";
}

//--------------------------------------------------------- GPU Code Executor -------------------------------------------------------------/

GpuCodeExecutor::GpuCodeExecutor(LpuBatchController *lpuBatchController, int distinctPpuCount) {
	this->lpuBatchController = lpuBatchController;
	this->distinctPpuCount = distinctPpuCount;
}

void GpuCodeExecutor::submitNextLpu(LPU *lpu, int ppuGroupIndex) {

	if (lpuBatchController->canAddNewLpu() && lpuBatchController->canHoldLpu(lpu)) {
		if (lpuBatchController->isEmptyBatch()) {
			lpuBatchRangeVector->at(ppuGroupIndex) = Range(lpu->id);
		} else {
			lpuBatchRangeVector->at(ppuGroupIndex).max++;
		}
		lpuBatchController->addLpuToTheCurrentBatch(lpu, ppuGroupIndex);
		return;	
	}
	if (!lpuBatchController->isEmptyBatch()) {
		execute();
	}

	lpuBatchController->addLpuToTheCurrentBatch(lpu, ppuGroupIndex);
	lpuBatchRangeVector->at(ppuGroupIndex) = Range(lpu->id);
}

void GpuCodeExecutor::forceExecution() {
	if (!lpuBatchController->isEmptyBatch()) {
		execute();
	}
}

void GpuCodeExecutor::submitNextLpus(std::vector<LPU*> *lpuVector) {
	for (unsigned int i = 0; i < lpuVector->size(); i++) {
		LPU *lpu = lpuVector->at(i);
		if (lpu != NULL) {
			int ppuGroupIndex = i % distinctPpuCount;
			submitNextLpu(lpu, ppuGroupIndex);
		}
	}
}

void GpuCodeExecutor::execute() {
	
	struct timeval tv;
        gettimeofday(&tv, NULL);
        long startTime = tv.tv_sec * 1000000 + tv.tv_usec;

	lpuBatchController->submitCurrentBatchToGpu();

	gettimeofday(&tv, NULL);
        long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	offloadStats->addStagingInTime(timeTaken);

	startTime = endTime;
	
	offloadFunction();
	cudaThreadSynchronize();
	check_error(cudaGetLastError(), *logFile);
	
	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	offloadStats->addExecutionTime(timeTaken);

	startTime = endTime;

	lpuBatchController->updateBatchDataPartsFromGpuResults();
	lpuBatchController->resetController();

	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	offloadStats->addStagingOutTime(timeTaken);

	resetCurrentBatchLpuRanges();
}

void GpuCodeExecutor::initialize() {
	this->offloadStats = new OffloadStats();
	lpuBatchRangeVector = new std::vector<Range>;
	lpuBatchRangeVector->reserve(distinctPpuCount);
	resetCurrentBatchLpuRanges();
}

void GpuCodeExecutor::cleanup() { 
	cudaDeviceReset(); 
	offloadStats->describe(*logFile);
	delete lpuCountVector;
	delete lpuBatchRangeVector;
}

void GpuCodeExecutor::resetCurrentBatchLpuRanges() {
	lpuBatchRangeVector->clear();
	for (int i = 0; i < distinctPpuCount; i++) {
		lpuBatchRangeVector->push_back(Range(INVALID_ID));
	} 
}
