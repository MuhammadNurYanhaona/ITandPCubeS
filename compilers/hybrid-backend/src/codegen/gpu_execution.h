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

/*---------------------------------------------------------------------------------------------------------- 
		    Functions related to communicating Metadata from Host to GPU
----------------------------------------------------------------------------------------------------------*/

// To construct LPUs from their IDs and regenarate dimensionality and other information inside the GPU, we 
// need to make available a minimum set of LPU related metadata to the GPU kernels. This function generate 
// a structure that holds that minimum set. The structure is LPS specific as our so far understanding is 
// the metadata for LPU reconstruction is the same regardless of the offloading context as long as the LPS
// is the same. 
void generateOffloadingMetadataStruct(Space *gpuContextLps, std::ofstream &headerFile);

// When distinction made between the SMs/Warps of the GPU from the host in a particular offloading context,
// then the LPUs offloaded to them also have diverging partition hierarchy that result in variation in 
// their metadata. So an aggregator structure is needed that will have one instance of the structure gene-
// rated by the previous function for each SM/Warp. Since the PPU count is a static value once the mapping
// configuration is known, it is possible to have an aggregator structure with a static array of metadata.
void generateMetadataAggregatorStruct(Space *gpuContextLps, 
		PCubeSModel *pcubesModel,
                std::ofstream &headerFile);

// This function generates a routine that creates and initializes an LPU reconstruction aggregate metadata
// structure from the properties of an GPU-Code-Executor
void generateKernelLaunchMatadataStructFn(Space *gpuContextLps,
                PCubeSModel *pcubesModel,
                const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile);

// Unlike in conventional CUDA programming model, we cannot assume any fixed size for the SM local shared
// memory version of data parts. This is because the programmer choose runtime arguments for the partition
// functions that determine what are the data part sizes. Therefore, we have to use dynamic shared memory
// and do memory allocation from that dynamic memory for different data parts explicitly. This function
// generate a metadata structure that holds information regarding the largest part sizes for different data
// structures that will be used as part of a batch LPU execution so that dynamic shared memory requirements
// can be determined properly. 
void generateMaxPartSizeMetadataStruct(GpuExecutionContext *gpuContext, std::ofstream &headerFile);	

// This function calls the above functions to generate metadata structures for all GPU contexts of a task
void generateAllLpuMetadataStructs(List<GpuExecutionContext*> *gpuExecutionContextList,
		PCubeSModel *pcubesModel,
                const char *initials,
                const char *headerFile, const char *programFile);

/*---------------------------------------------------------------------------------------------------------- 
			Functions related to generating LPU Batch Controllers
----------------------------------------------------------------------------------------------------------*/

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

/*---------------------------------------------------------------------------------------------------------- 
			  Functions related to generating GPU Code Executors
----------------------------------------------------------------------------------------------------------*/

// GPU code executors serve as the broker between host and the GPU computation for a particular offloading
// context. A GPU code executor has a reference to a context specific LpuBatchController for staging in and
// out data and it embodies an offloading funcion that invokes the CUDA kernel(s) to simulate the logic of
// the sub-flow represented by the underlying GPU execution context. This function generates the definition
// of a GPU code executor
void generateGpuCodeExecutorForContext(GpuExecutionContext *gpuContext,
		PCubeSModel *pcubesModel,
                const char *initials,
                std::ofstream &headerFile, std::ofstream &programFile);

// This uses the function above to generate GPU code executors for all GPU execution contexts of the current
// task
void generateGpuCodeExecutors(List<GpuExecutionContext*> *gpuExecutionContextList, 
		PCubeSModel *pcubesModel,
		const char *initials, 
		const char *headerFile, const char *programFile);

// these are four auxiliary functions to be used by the Gpu-Code-Executor generator function of the above to
// provide implementations for the constructor and some inherited functions
void generateGpuCodeExecutorConstructor(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile);
void generateGpuCodeExecutorInitializer(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile);
void generateGpuCodeExecutorOffloadFn(GpuExecutionContext *gpuContext,
		PCubeSModel *pcubesModel, 
		const char *initials, std::ofstream &programFile);
void generateGpuCodeExecutorCleanupFn(GpuExecutionContext *gpuContext, 
		const char *initials, std::ofstream &programFile);

// these two functions are used to generate kernels related to the offloading context a GPU-Code-Executor is
// responsible for
void generateGpuCodeExecutorKernel(CompositeStage *kernelDef,
		GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile);
void generateGpuCodeExecutorKernelList(GpuExecutionContext *gpuContext,
                PCubeSModel *pcubesModel,
                const char *initials, std::ofstream &programFile);

/*---------------------------------------------------------------------------------------------------------- 
			    Host and GPU brokerage functions generators
----------------------------------------------------------------------------------------------------------*/

// The batch-mode PPU controller needs references of all the GPU code executor classes to be able to offload 
// a GPU computation to specific executor when the need may arise. This routine generates a function that
// instantiates all GPU code executors of a class and puts them in a map searchable by the context ID. 
void generateGpuExecutorMapFn(List<GpuExecutionContext*> *gpuExecutionContextList,
                const char *initials,
                const char *headerFile, const char *programFile);

#endif
