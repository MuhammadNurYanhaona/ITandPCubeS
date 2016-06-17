#ifndef _H_gpu_execution
#define _H_gpu_execution

/* This header file contains library routines that generate task specific and within task context specific 
   classes for offloading computations to GPU from host. Furthermore, it contains the  functions to set up 
   the CUDA program file and other GPU execution related features. 
*/

#include "space_mapping.h"
#include "../semantics/task_space.h"
#include "../static-analysis/gpu_execution_ctxt.h"
#include "../utils/list.h"

#include <fstream>

// this function put the proper includes directives in the CUDA program file
void initializeCudaProgramFile(const char *initials, const char *headerFile, const char *programFile);

void generateBatchConfigurationConstants(const char *headerFile, PCubeSModel *pcubesModel);

/*--------------------------------------------------------------------------------------------------------- 
			Functions related to generating LPU Batch Controllers
---------------------------------------------------------------------------------------------------------*/

// LPU batch controllers do the memory allocations and data stage-in and out in both end of the computation
// offloading. This function generates a context specific LPU batch controller 
void generateLpuBatchControllerForContext(GpuExecutionContext *gpuContext, 
		PCubeSModel *pcubesModel,
		const char *initials, 
		std::ofstream &headerFile, std::ofstream &programFile);

// this uses the above function to generate LPU batch controllers for all sub-flows found in the task that
// should be executed in the GPU
void generateLpuBatchControllers(List<GpuExecutionContext*> *gpuExecutionContextList, 
		PCubeSModel *pcubesModel,
		const char *initials, 
		const char *headerFile, const char *programFile);

// these three are supporting routines used by the generateLpuBatchControllerForLps routine that provide 
// implementations for three functions of an LPU batch controller
void generateLpuBatchControllerConstructor(GpuExecutionContext *gpuContext, 
		PCubeSModel *pcubesModel,
		const char *initials, std::ofstream &programFile);
void generateLpuBatchControllerLpuAdder(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile);
void generateLpuBatchControllerMemchecker(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile);

/*--------------------------------------------------------------------------------------------------------- 
			  Functions related to generating GPU Code Executors
---------------------------------------------------------------------------------------------------------*/

#endif
