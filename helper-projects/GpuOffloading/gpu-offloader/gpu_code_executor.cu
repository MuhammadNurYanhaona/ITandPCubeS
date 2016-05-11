#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

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
	
	lpuBatchController->submitCurrentBatchToGpu();
	
	offloadFunction();
	cudaThreadSynchronize();
	check_error(cudaGetLastError(), *logFile);

	lpuBatchController->updateBatchDataPartsFromGpuResults();
	lpuBatchController->resetController();
}

void GpuCodeExecutor::initialize() {}

void GpuCodeExecutor::cleanup() { cudaDeviceReset(); }
