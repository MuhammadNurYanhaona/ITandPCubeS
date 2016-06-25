#include "gpu_code_executor.h"
#include "lpu_parts_tracking.h"
#include "../runtime/structure.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <vector>

/******************************************************************************************************************************************* 
			Note that *_cuda.cu files have the code that needs the NVCC compiler for compilation
*******************************************************************************************************************************************/

//--------------------------------------------------------- GPU Code Executor -------------------------------------------------------------/

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

void GpuCodeExecutor::cleanup() { 
	cudaDeviceReset(); 
	offloadStats->describe(*logFile);
	delete lpuCountVector;
	delete lpuBatchRangeVector;
}
