#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"

#include <sys/time.h>
#include <vector>

//-------------------------------------------------------- Offload Statistics -------------------------------------------------------------/

OffloadStats::OffloadStats() {
	timeSpentDataPartsPreparation = 0;
	timeSpentStagingIn = 0;
	timeSpentExecution = 0;
	timeSpentStagingOut = 0;
	executionCount = 0;
}

void OffloadStats::describe(std::ofstream &logFile) {
	logFile << "Overall time spent preparing data parts in the CPU: ";
	logFile << timeSpentDataPartsPreparation << " Seconds\n";
	logFile << "Overall time spent staging data into GPU from CPU: ";
	logFile << timeSpentStagingIn << " Seconds\n";
	logFile << "Overall time spent executing kernels for the LPUs: ";
	logFile << timeSpentExecution << " Seconds\n";
	logFile << "Overall time spent staging data out from GPU to CPU: ";
	logFile << timeSpentStagingOut << " Seconds\n";
	logFile << "Number of time the offloading function has been Executed: ";
	logFile << executionCount << "\n";	
}

//--------------------------------------------------------- GPU Code Executor -------------------------------------------------------------/

GpuCodeExecutor::GpuCodeExecutor(LpuBatchController *lpuBatchController, int distinctPpuCount) {
	this->lpuBatchController = lpuBatchController;
	this->distinctPpuCount = distinctPpuCount;
}

void GpuCodeExecutor::submitNextLpu(LPU *lpu, int ppuGroupIndex) {

	if (lpuBatchController->canAddNewLpu() && lpuBatchController->canHoldLpu(lpu)) {
                Range lpuIdRange = lpuBatchRangeVector->at(ppuGroupIndex);
                if (lpuIdRange.min == INVALID_ID) {
			extractAncestorLpuConfigs(lpu, ppuGroupIndex);
                        lpuBatchRangeVector->at(ppuGroupIndex) = Range(lpu->id);
                } else {
                        lpuBatchRangeVector->at(ppuGroupIndex).max++;
                }

		struct timeval tv;
        	gettimeofday(&tv, NULL);
        	long startTime = tv.tv_sec * 1000000 + tv.tv_usec;
		lpuBatchController->addLpuToTheCurrentBatch(lpu, ppuGroupIndex);
		gettimeofday(&tv, NULL);
		long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
		double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
		offloadStats->addDataPartsPreparationTime(timeTaken);

		return;	
	}
	if (!lpuBatchController->isEmptyBatch()) {
		execute();
	}

	struct timeval tv;
	gettimeofday(&tv, NULL);
	long startTime = tv.tv_sec * 1000000 + tv.tv_usec;
	lpuBatchController->addLpuToTheCurrentBatch(lpu, ppuGroupIndex);
	gettimeofday(&tv, NULL);
	long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	offloadStats->addDataPartsPreparationTime(timeTaken);
	
	extractAncestorLpuConfigs(lpu, ppuGroupIndex);
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

void GpuCodeExecutor::initialize() {
	this->offloadStats = new OffloadStats();
	lpuBatchRangeVector = new std::vector<Range>;
	lpuBatchRangeVector->reserve(distinctPpuCount);
	resetCurrentBatchLpuRanges();
}

void GpuCodeExecutor::resetCurrentBatchLpuRanges() {
	lpuBatchRangeVector->clear();
	for (int i = 0; i < distinctPpuCount; i++) {
		lpuBatchRangeVector->push_back(Range(INVALID_ID));
	} 
}
