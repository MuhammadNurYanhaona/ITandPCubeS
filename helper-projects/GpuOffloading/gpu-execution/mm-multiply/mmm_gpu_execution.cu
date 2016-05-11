#include <cuda.h>
#include <cuda_runtime.h>

#include "mmm_gpu_execution.h"
#include "../../test-case/mmm_structure.h"
#include "../../runtime/structure.h"
#include "../../gpu-utils/gpu_constant.h"
#include "../../gpu-offloader/gpu_code_executor.h"
#include "../../gpu-offloader/lpu_parts_tracking.h"
#include "../../utils/list.h"
#include "../../gpu-utils/gpu_partition.h"
#include "../../gpu-utils/gpu_utils.h"

//----------------------------------------------------- MMM Batch LPU Controller -----------------------------------------------------/

MMMLpuBatchController::MMMLpuBatchController(int lpuCountThreshold, long memLimit)  : LpuBatchController() {
	
	List<const char*> *propertyNames = new List<const char*>;
	propertyNames->Append("a");
	propertyNames->Append("b");
	propertyNames->Append("c");
	
	List<const char*> *toBeModifiedProperties = new List<const char*>;
	toBeModifiedProperties->Append("c");

	initialize(lpuCountThreshold, memLimit, propertyNames, toBeModifiedProperties);		
}

int MMMLpuBatchController::calculateLpuMemoryRequirement(LPU *lpu) {
	MMMLpu *mmmLpu = (MMMLpu *) lpu;
	int size = 0;
	if (!dataPartTracker->isAlreadyIncluded(mmmLpu->aPartId, "a")) {
		size += (mmmLpu->aPartDims[0].storage.getLength() 
			* mmmLpu->aPartDims[1].storage.getLength()) * sizeof(double);
	}
	if (!dataPartTracker->isAlreadyIncluded(mmmLpu->bPartId, "b")) {
		size += (mmmLpu->bPartDims[0].storage.getLength() 
			* mmmLpu->bPartDims[1].storage.getLength()) * sizeof(double);
	}
	if (!dataPartTracker->isAlreadyIncluded(mmmLpu->aPartId, "c")) {
		size += (mmmLpu->cPartDims[0].storage.getLength() 
			* mmmLpu->cPartDims[1].storage.getLength()) * sizeof(double);
	}
	return size;
}

void MMMLpuBatchController::addLpuToTheCurrentBatch(LPU *lpu) {
	
	MMMLpu *mmmLpu = (MMMLpu *) lpu;

	LpuDataPart *aPart = new LpuDataPart(2, 
			mmmLpu->aPartDims, mmmLpu->a, sizeof(double), mmmLpu->aPartId);
	bool notRedundant = dataPartTracker->addDataPart(aPart, "a");
	if (!notRedundant) {
		delete aPart;
	}
	LpuDataPart *bPart = new LpuDataPart(2, 
			mmmLpu->bPartDims, mmmLpu->b, sizeof(double), mmmLpu->bPartId);
	notRedundant = dataPartTracker->addDataPart(bPart, "b");
	if (!notRedundant) {
		delete bPart;
	}
	LpuDataPart *cPart = new LpuDataPart(2, 
			mmmLpu->cPartDims, mmmLpu->c, sizeof(double), mmmLpu->cPartId);
	notRedundant = dataPartTracker->addDataPart(cPart, "c");
	if (!notRedundant) {
		delete cPart;
	}

	LpuBatchController::addLpuToTheCurrentBatch(lpu);
}

//------------------------------------------------------ Offloading GPU Kernels ------------------------------------------------------/

__global__ void matrixMultiplyKernel(MMMLpuBatchRange batchRange, 
		mmm::Partition partition, 
		mmm::ArrayMetadata arrayMetadata,
		mmm::TaskGlobals *taskGlobals,
		mmm::ThreadLocals *threadLocals,	
		GpuBufferReferences aBuffers, 
		GpuBufferReferences bBuffers, 
		GpuBufferReferences cBuffers) {

	/*----------------------------------------------------------------------------------------------------------------------------
						    Space A: Top-most User Defined Space 
	----------------------------------------------------------------------------------------------------------------------------*/

	// before we can do anything in the kernel, we need to determine the thread, warp, and sm IDs of the thread
	// executing the kernel code
        int smId = blockIdx.x;
        int warpId = threadIdx.x / WARP_SIZE;
	int threadId = threadIdx.x % WARP_SIZE;
	
	// variables for holding the data part references for the top-space LPU
	double *a, *b, *c;

	// variables for tracking storage and partition dimensions of the top space LPU's data parts
	__shared__ int aSRanges[2][2], bSRanges[2][2], cSRanges[2][2];
	__shared__ int aPRanges[2][2], bPRanges[2][2], cPRanges[2][2];

	// SMs stride over different indexes to get different LPUs to operate on
	Range lpuIdRange = batchRange.lpuIdRange;
	for (int linearId = lpuIdRange.min + smId; linearId <= lpuIdRange.max; linearId += BLOCK_COUNT) {

		// point the a, b, c matrix references to the memory addresses where corresponding data parts for the
		// current LPUs starts 

		if (warpId == 0 && threadId == 0) {
			//------------------------------------------------------------- retrieve a and its dimensions
			int lpuIndex = linearId - lpuIdRange.min;
			int aIndex = aBuffers.partIndexBuffer[lpuIndex];
			int aStartsAt = aBuffers.partBeginningBuffer[aIndex];
			a = (double *) aBuffers.dataBuffer[aStartsAt];
			int aDimRangeStart = lpuIndex * 2 * 2 * 2; // there are storage and partition ranges each has
								   // two integers and the data structure is 2D
			aSRanges[0][0] = aBuffers.partBeginningBuffer[aDimRangeStart];
			aSRanges[0][1] = aBuffers.partBeginningBuffer[aDimRangeStart + 1];
			aSRanges[1][0] = aBuffers.partBeginningBuffer[aDimRangeStart + 2];
			aSRanges[1][1] = aBuffers.partBeginningBuffer[aDimRangeStart + 3];

			aPRanges[0][0] = aBuffers.partBeginningBuffer[aDimRangeStart + 4];
			aPRanges[0][1] = aBuffers.partBeginningBuffer[aDimRangeStart + 5];
			aPRanges[1][0] = aBuffers.partBeginningBuffer[aDimRangeStart + 6];
			aPRanges[1][1] = aBuffers.partBeginningBuffer[aDimRangeStart + 7];
			
			//------------------------------------------------------------- retrieve b and its dimensions
			int bIndex = bBuffers.partIndexBuffer[lpuIndex];
			int bStartsAt = bBuffers.partBeginningBuffer[bIndex];
			b = (double *) bBuffers.dataBuffer[bStartsAt];
			int bDimRangeStart = lpuIndex * 2 * 2 * 2; 
			bSRanges[0][0] = bBuffers.partBeginningBuffer[bDimRangeStart];
			bSRanges[0][1] = bBuffers.partBeginningBuffer[bDimRangeStart + 1];
			bSRanges[1][0] = bBuffers.partBeginningBuffer[bDimRangeStart + 2];
			bSRanges[1][1] = bBuffers.partBeginningBuffer[bDimRangeStart + 3];

			bPRanges[0][0] = bBuffers.partBeginningBuffer[bDimRangeStart + 4];
			bPRanges[0][1] = bBuffers.partBeginningBuffer[bDimRangeStart + 5];
			bPRanges[1][0] = bBuffers.partBeginningBuffer[bDimRangeStart + 6];
			bPRanges[1][1] = bBuffers.partBeginningBuffer[bDimRangeStart + 7];
			
			//------------------------------------------------------------- retrieve c and its dimensions
			int cIndex = cBuffers.partIndexBuffer[lpuIndex];
			int cStartsAt = cBuffers.partBeginningBuffer[cIndex];
			c = (double *) cBuffers.dataBuffer[cStartsAt];
			int cDimRangeStart = lpuIndex * 2 * 2 * 2; 
			cSRanges[0][0] = cBuffers.partBeginningBuffer[cDimRangeStart];
			cSRanges[0][1] = cBuffers.partBeginningBuffer[cDimRangeStart + 1];
			cSRanges[1][0] = cBuffers.partBeginningBuffer[cDimRangeStart + 2];
			cSRanges[1][1] = cBuffers.partBeginningBuffer[cDimRangeStart + 3];
			
			cPRanges[0][0] = cBuffers.partBeginningBuffer[cDimRangeStart + 4];
			cPRanges[0][1] = cBuffers.partBeginningBuffer[cDimRangeStart + 5];
			cPRanges[1][0] = cBuffers.partBeginningBuffer[cDimRangeStart + 6];
			cPRanges[1][1] = cBuffers.partBeginningBuffer[cDimRangeStart + 7];	
		}
		__syncthreads();

		/*--------------------------------------------------------------------------------------------------------------------
						Space A-Sub: Compiler Generated Space for Subpartition 
		--------------------------------------------------------------------------------------------------------------------*/

		// once we have the storage and partition dimensions of data structure at the top-level space's LPU
		// we can determine the sub-partition space's Lpu count
		int subpartitionCount = block_size_part_count(bPRanges[0], partition.blockSize);
		
		__shared__ int aPSubRanges[2][2], bPSubRanges[2][2];

		// the subpartitioned LPUs are processed one by one; remember that LPUs of sub-partitioned LPSes are
		// not supposed to be distributed
		for (int subLpu = 0; subLpu < subpartitionCount; subLpu++) {
			
			if (warpId == 0 && threadId == 0) {	
				// first we need to determine the partition dimension ranges of the two sub-
				// partitioned data structures, which are matrix A and B
				int lpuId = subLpu;
				aPSubRanges[0][0] = aPRanges[0][0];
				aPSubRanges[0][1] = aPRanges[0][1];
				block_size_part_range(aPSubRanges[1], aPRanges[1],
						subpartitionCount, lpuId, partition.blockSize, 0, 0); 
				block_size_part_range(bPSubRanges[0], bPRanges[0],
						subpartitionCount, lpuId, partition.blockSize, 0, 0); 
				bPSubRanges[1][0] = bPRanges[1][0];
				bPSubRanges[1][1] = bPRanges[1][1];
			}
			__syncthreads();

			// here we should load sub-section of A and B from the GPU card memory to the local memory
			// what about C? Or should we directly perform all computation on the card memory and rely
			// on the hardware's caching machanism to do the global and shared memory interactions?
		
			// In the multicore and segmented memory architecture cases the matrix-matrix multiplication 
			// code starts here. In the GPU, the existing partition scheme will result in only one warp 
			// within an SM doing computation for the user code. Rather the user should have the 
			// computation to be distributed to multiple warps for different smaller sub-sections of the 
			// block of matrix C using another lower level LPS

			/*------------------------------------------------------------------------------------------------------------
							   Space B: Lowest User Defined Space 
			------------------------------------------------------------------------------------------------------------*/

			// Space B LPUs will be distributed among the warps; so the parts' dimension configuration
			// should be different for different warps and we cannot have a single shared object per
			// part information as we have done in the previous LPSes. Rather, we will have a shared
			// memory pannel having 1 entry per warp to hold relevant part's dimension configuration.
			__shared__ int aSpaceBPRanges[WARP_COUNT][2][2];
			__shared__ int bSpaceBPRanges[WARP_COUNT][2][2];
			__shared__ int cSpaceBPRanges[WARP_COUNT][2][2];

			int spaceBLpuCount1 = block_size_part_count(cPRanges[0], 1);
			int spaceBLpuCount2 = block_size_part_count(cPRanges[1], partition.blockSize);
			int spaceBLpuCount = spaceBLpuCount1 * spaceBLpuCount2;

			// distribute the Space B LPUs among the warps
			for (int spaceBLpu = warpId; spaceBLpu < spaceBLpuCount; spaceBLpu += WARP_COUNT) {
				
				if (threadId == 0) {
					// construct the 2 dimensional LPU ID from the linear LPU Id
					int spaceBLpuId[2];
					spaceBLpuId[0] = spaceBLpu / spaceBLpuCount2;
					spaceBLpuId[1] = spaceBLpu % spaceBLpuCount2;
					
					//---------------------------------------------------- A part dimensions
					block_size_part_range(aSpaceBPRanges[warpId][0], aPSubRanges[0],
							spaceBLpuCount1, spaceBLpuId[0], 1, 0, 0); 
					aSpaceBPRanges[warpId][1][0] = aPSubRanges[1][0];
					aSpaceBPRanges[warpId][1][1] = aPSubRanges[1][1];

					//---------------------------------------------------- B part dimensions
					bSpaceBPRanges[warpId][0][0] = bPSubRanges[0][0];
					bSpaceBPRanges[warpId][0][1] = bPSubRanges[0][1];
					block_size_part_range(bSpaceBPRanges[warpId][1], bPSubRanges[1],
							spaceBLpuCount2, spaceBLpuId[1], 1, 0, 0); 
					
					//---------------------------------------------------- C part dimensions
					block_size_part_range(cSpaceBPRanges[warpId][0], cPRanges[0],
							spaceBLpuCount1, spaceBLpuId[0], 1, 0, 0); 
					block_size_part_range(cSpaceBPRanges[warpId][1], cPRanges[1],
							spaceBLpuCount2, spaceBLpuId[1], 1, 0, 0);
				} 
				// there is no syncthread operation needed here as updates done by a thread in a
				// warp is visible to all other threads in that warp
					
				/*----------------------------------------------------------------------------------------------------
								  Translated Computation Stage 
				----------------------------------------------------------------------------------------------------*/

				// the compute stage for IT matrix-matrix multiplication looks like the following
				// do { 
				// 	c[i][j] = c[i][j] +  a[i][k] * b[k][j]
				// } for i, j in c; k in a
				// In each warp we have 32 threads performing the same instruction in a lock-step
				// fasion. If we can make the threads working on different piece of data then we
				// can have a vectorized translation of the IT for loop without any additional data
				// synchronization among the threads. A simple static analysis of the code block 
				// should detect that i and j indices both appeared on the left hand side of the
				// enclosed statement but not the k index. So we can let different threads work on
				// different i or j values. In general, we should avoid varying both indices at the
				// same time to reduce memory bank conflicts.
				
				// But how do we select the index for distribution among threads that has the best 
				// potential for coalescing global memory and reducing shared memory accesses? The 
				// selection also need be cautious about compromising opportunities of parallelism.
				// The initial solution for this is incorporated in GPU utility library that, given,
				// a set of ranges to iterate, provides loop starting indexes and step sizes.   
				int iterableRanges[4];
				iterableRanges[0] = cSpaceBPRanges[warpId][0][0];
				iterableRanges[1] = cSpaceBPRanges[warpId][0][1];
				iterableRanges[2] = cSpaceBPRanges[warpId][1][0];
				iterableRanges[3] = cSpaceBPRanges[warpId][1][1];
				int indexesAndSteps[4];
				determineLoopIndexesAndSteps(2, threadId, iterableRanges, indexesAndSteps); 
				
				// iterate over the rows
				int iStart = indexesAndSteps[0];
				int iEnd = iterableRanges[1];
				int iStep = indexesAndSteps[1];
				for (int i = iStart; i <= iEnd; i += iStep) {

					int c_i = i - cSRanges[0][0];
					int a_i = i - aSRanges[0][0];
				
					// iterate over the columns
					int jStart = indexesAndSteps[2];
					int jEnd = iterableRanges[3];
					int jStep = indexesAndSteps[3];
					for (int j = jStart; j <= jEnd; j+= jStep) {
	
						int c_j = j - cSRanges[1][0];
						int b_j = j - bSRanges[1][0];
						
						// iterate over the common dimension
						int kStart = aSpaceBPRanges[warpId][1][0];
						int kEnd = aSpaceBPRanges[warpId][1][1];
						for (int k = kStart; k <= kEnd; k++) {
							
							int a_k = k - aSRanges[1][0];
							int b_k = k - bSRanges[0][0];

							int cIndex = c_i * (cSRanges[1][1] - cSRanges[1][0] + 1) + c_j;
							int aIndex = a_i * (aSRanges[1][1] - aSRanges[1][0] + 1) + a_k;
							int bIndex = b_k * (bSRanges[1][1] - bSRanges[1][0] + 1) + b_j;

							c[cIndex] += a[aIndex] * b[bIndex];	
						}
					}
				}
			} 
		}
	}
}


//------------------------------------------------------- MMM GPU Code Executor ------------------------------------------------------/

MMMGpuCodeExecutor::MMMGpuCodeExecutor(LpuBatchController *lpuBatchController, 
		mmm::Partition partition, 
		mmm::ArrayMetadata arrayMetadata,
		mmm::TaskGlobals *taskGlobals,
		mmm::ThreadLocals *threadLocals) 
		: GpuCodeExecutor(lpuBatchController) {

	this->partition = partition;
	this->arrayMetadata = arrayMetadata;
	this->taskGlobalsCpu = taskGlobals;
	this->taskGlobalsGpu = NULL;
	this->threadLocalsCpu = threadLocals;
	this->threadLocalsGpu = NULL;
}

void MMMGpuCodeExecutor::offloadFunction() {
	
	GpuBufferReferences aBuffers = lpuBatchController->getGpuBufferReferences("a");
	GpuBufferReferences bBuffers = lpuBatchController->getGpuBufferReferences("b");
	GpuBufferReferences cBuffers = lpuBatchController->getGpuBufferReferences("c");

	MMMLpuBatchRange batchRange;
	batchRange.lpuIdRange = currentBatchLpuRange;
	batchRange.lpuCount1 = lpuCount[0];
	batchRange.lpuCount2 = lpuCount[1];

	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
	matrixMultiplyKernel<<< BLOCK_COUNT, threadsPerBlock >>>(batchRange,
			partition, arrayMetadata, 
			taskGlobalsGpu, threadLocalsGpu, 
			aBuffers, bBuffers, cBuffers);
}

void MMMGpuCodeExecutor::initialize() {

	size_t taskGlobalsSize = sizeof(taskGlobalsCpu);
	cudaMalloc((void **) &taskGlobalsGpu, taskGlobalsSize);
	cudaMemcpy(taskGlobalsGpu, taskGlobalsCpu, taskGlobalsSize, cudaMemcpyHostToDevice);

	size_t threadLocalsSize = sizeof(threadLocalsCpu);
	cudaMalloc((void **) &threadLocalsGpu, threadLocalsSize);
	cudaMemcpy(threadLocalsGpu, threadLocalsCpu, threadLocalsSize, cudaMemcpyHostToDevice);
}
        
void MMMGpuCodeExecutor::cleanup() {

	size_t taskGlobalsSize = sizeof(taskGlobalsCpu);
	cudaMemcpy(taskGlobalsCpu, taskGlobalsGpu, taskGlobalsSize, cudaMemcpyDeviceToHost);
	size_t threadLocalsSize = sizeof(threadLocalsCpu);
	cudaMemcpy(threadLocalsCpu, threadLocalsGpu, threadLocalsSize, cudaMemcpyDeviceToHost);

	GpuCodeExecutor::cleanup();
}

