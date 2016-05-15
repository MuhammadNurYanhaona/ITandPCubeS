#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

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
	*logFile << ((endTime - startTime)) / (1000 * 1000) << " seconds\n";
	logFile->flush();

	startTime = endTime;
	
	offloadFunction();
	cudaThreadSynchronize();
	check_error(cudaGetLastError(), *logFile);
	
	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	*logFile << "\tGPU kernel(s) execution time: ";
	*logFile << ((endTime - startTime)) / (1000 * 1000) << " seconds\n";
	logFile->flush();

	startTime = endTime;

	lpuBatchController->updateBatchDataPartsFromGpuResults();
	lpuBatchController->resetController();

	gettimeofday(&tv, NULL);
        endTime = tv.tv_sec * 1000000 + tv.tv_usec;
	*logFile << "\tGPU to CPU data synchronization tme: ";
	*logFile << ((endTime - startTime)) / (1000 * 1000) << " seconds\n";

	*logFile << "Finished executing a batch\n";
	logFile->flush();
}

void GpuCodeExecutor::initialize() {}

void GpuCodeExecutor::cleanup() { cudaDeviceReset(); }
