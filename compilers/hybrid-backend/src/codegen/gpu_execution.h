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

// this function put the proper includes in the CUDA program file
void initializeCudaProgramFile(const char *initials, const char *headerFile, const char *programFile);

void generateBatchConfigurationConstants(const char *headerFile, PCubeSModel *pcubesModel);

// LPU batch controllers do the memory allocations and data stage-in and out in both end of the computation
// offloading. This function generates an LPS specific LPU batch controller 
void generateLpuBatchControllerForLps(Space *gpuContextLps, const char *initials, 
		std::ofstream &headerFile, std::ofstream &programFile);

// this uses the above function to generate LPU batch controllers for all sub-flows found in the task that
// should be executed in the GPU
void generateLpuBatchControllers(List<GpuExecutionContext*> *gpuExecutionContextList, 
		const char *initials, 
		const char *headerFile, const char *programFile);

#endif
