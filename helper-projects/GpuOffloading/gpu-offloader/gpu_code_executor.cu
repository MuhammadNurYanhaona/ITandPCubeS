#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

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

GpuCodeExecutor::GpuCodeExecutor(LpuBatchController *lpuBatchController) {
	this->lpuBatchController = lpuBatchController;
}

void GpuCodeExecutor::submitNextLpu(LPU *lpu) {

	if (lpuBatchController->canAddNewLpu() && lpuBatchController->canHoldLpu(lpu)) {
		if (lpuBatchController->isEmptyBatch()) {
			currentBatchLpuRange = Range(lpu->id);
		} else {
			currentBatchLpuRange.max++;
		}
		lpuBatchController->addLpuToTheCurrentBatch(lpu);
		return;	
	}
	if (!lpuBatchController->isEmptyBatch()) {
		execute();
	}

	lpuBatchController->addLpuToTheCurrentBatch(lpu);
	currentBatchLpuRange = Range(lpu->id);
}

void GpuCodeExecutor::forceExecution() {
	if (!lpuBatchController->isEmptyBatch()) {
		execute();
	}
}

void GpuCodeExecutor::execute() {
	
	*logFile << "Going to execute a batch\n";
	logFile->flush();
	
	struct timeval tv;
        gettimeofday(&tv, NULL);
        long startTime = tv.tv_sec * 1000000 + tv.tv_usec;

	lpuBatchController->submitCurrentBatchToGpu();

	gettimeofday(&tv, NULL);
        long endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	*logFile << "\tCPU to GPU data copying time: ";
	double timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	*logFile << timeTaken << " seconds\n";
	logFile->flush();
	offloadStats->addStagingInTime(timeTaken);

	startTime = endTime;
	
	offloadFunction();
	cudaThreadSynchronize();
	check_error(cudaGetLastError(), *logFile);
	
	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	*logFile << "\tGPU kernel(s) execution time: ";
	timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	*logFile << timeTaken << " seconds\n";
	logFile->flush();
	offloadStats->addExecutionTime(timeTaken);

	startTime = endTime;

	lpuBatchController->updateBatchDataPartsFromGpuResults();
	lpuBatchController->resetController();

	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	*logFile << "\tGPU to CPU data synchronization tme: ";
	timeTaken = ((endTime - startTime) * 1.0) / (1000 * 1000);
	*logFile << timeTaken << " seconds\n";
	offloadStats->addStagingOutTime(timeTaken);

	*logFile << "Finished executing a batch\n";
	logFile->flush();
}

void GpuCodeExecutor::initialize() {
	this->offloadStats = new OffloadStats();
}

void GpuCodeExecutor::cleanup() { 
	cudaDeviceReset(); 
	offloadStats->describe(*logFile);
}
