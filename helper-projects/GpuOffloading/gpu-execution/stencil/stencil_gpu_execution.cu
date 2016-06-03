#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "stencil_gpu_execution.h"
#include "../../test-case/stencil/stencil_structure.h"
#include "../../runtime/structure.h"
#include "../../gpu-utils/gpu_constant.h"
#include "../../gpu-offloader/gpu_code_executor.h"
#include "../../gpu-offloader/lpu_parts_tracking.h"
#include "../../utils/list.h"
#include "../../gpu-utils/gpu_partition.h"
#include "../../gpu-utils/gpu_utils.h"

//----------------------------------------------------- Stencil Batch LPU Controller -----------------------------------------------------/

StencilLpuBatchController::StencilLpuBatchController(int lpuCountThreshold, 
		long memConsumptionLimit) : LpuBatchController() {
	
	List<const char*> *versionlessProperties = new List<const char*>;
	List<const char*> *multiversionProperties = new List<const char*>;
	multiversionProperties->Append("plate");
	List<const char*> *propertyNames = new List<const char*>;
        propertyNames->AppendAll(versionlessProperties);
        propertyNames->AppendAll(multiversionProperties);

	setBufferManager(new LpuDataBufferManager(versionlessProperties, multiversionProperties));
        initialize(lpuCountThreshold, memConsumptionLimit, propertyNames, multiversionProperties);
}

int StencilLpuBatchController::calculateLpuMemoryRequirement(LPU *lpu) {
	
	stencil::StencilLpu *stencilLpu = (stencil::StencilLpu *) lpu;
	return 2 * (stencilLpu->platePartDims[0].storage.getLength()
                        * stencilLpu->platePartDims[1].storage.getLength()) * sizeof(double);
}

void StencilLpuBatchController::addLpuToTheCurrentBatch(LPU *lpu) {
	
	stencil::StencilLpu *stencilLpu = (stencil::StencilLpu *) lpu;
	List<void*> *versionList = new List<void*>;
	versionList->Append(stencilLpu->partReference->getData(0));
	versionList->Append(stencilLpu->partReference->getData(1));
	VersionedLpuDataPart *dataPart = new VersionedLpuDataPart(2, stencilLpu->platePartDims, 
			versionList, sizeof(double), stencilLpu->platePartId);
	dataPartTracker->addDataPart(dataPart, "plate");
	LpuBatchController::addLpuToTheCurrentBatch(lpu);
}

//---------------------------------------------------------- Offload Functions -----------------------------------------------------------/


__global__ void stencilKernel1(StencilLpuBatchRange batchRange,
                stencil::Partition partition,
                stencil::ArrayMetadata arrayMetadata,
                stencil::TaskGlobals *taskGlobals,
                stencil::ThreadLocals *threadLocals,
                VersionedGpuBufferReferences plateBuffers) {
	
	// before we can do anything in the kernel, we need to determine the thread, warp, and sm IDs of the thread
        // executing the kernel code
        int smId = blockIdx.x;
        int warpId = threadIdx.x / WARP_SIZE;
        int threadId = threadIdx.x % WARP_SIZE;

	/*----------------------------------------------------------------------------------------------------------------------------
                                                    		Space A Logic
        ----------------------------------------------------------------------------------------------------------------------------*/

	// variables holding the data part references for a particular top-level LPU for the entire SM
	__shared__ double *plate[2];
	__shared__ int plateSRanges[2][2], platePRanges[2][2];

	// since the plate is a multiversioned data structure whose part versions get updated as the SMs do computation
	// on them, an index is retained per LPU to keep track of the most up-to-date part version  
	__shared__ int plateVersionIndex;

	// SMs stride over different indexes to get different LPUs to operate on (Thus this kernel implements a mapping
	// where the upper level Space A is mapped to the SM and the lower level Space B is mapped to the warps)
        Range lpuIdRange = batchRange.lpuIdRange;
        for (int lpuId = lpuIdRange.min + smId; lpuId <= lpuIdRange.max; lpuId += BLOCK_COUNT) {
		
		// all threads should synchronize here to prevent the LPU metadata writer threads to overwrite the old
                // values before the other threads are done using those values
                __syncthreads();

		// locate the plate versions to appropriate location in the GPU card memory and set up the plate part's 
		// storage dimensions
		if (warpId == 0 && threadId == 0) {
			__shared__ int lpuIndex, plateIndex, plateStartsAt, plateDimRangeStart, plateSize;
                        lpuIndex = lpuId - lpuIdRange.min;
                        plateIndex = plateBuffers.partIndexBuffer[lpuIndex];
                        plateStartsAt = plateBuffers.partBeginningBuffer[plateIndex];
                        plateDimRangeStart = plateIndex * 2 * 2 * 2;

			// load storage range information in the shared memory
                        plateSRanges[0][0] = plateBuffers.partRangeBuffer[plateDimRangeStart];
                        plateSRanges[0][1] = plateBuffers.partRangeBuffer[plateDimRangeStart + 1];
                        plateSRanges[1][0] = plateBuffers.partRangeBuffer[plateDimRangeStart + 2];
                        plateSRanges[1][1] = plateBuffers.partRangeBuffer[plateDimRangeStart + 3];

			// load partition range information in the shared memory
                        platePRanges[0][0] = plateBuffers.partRangeBuffer[plateDimRangeStart + 4];
                        platePRanges[0][1] = plateBuffers.partRangeBuffer[plateDimRangeStart + 5];
                        platePRanges[1][0] = plateBuffers.partRangeBuffer[plateDimRangeStart + 6];
                        platePRanges[1][1] = plateBuffers.partRangeBuffer[plateDimRangeStart + 7];
                        
			// initialize plate data references
			plate[0] = (double *) (plateBuffers.dataBuffer + plateStartsAt);
			plateSize = (plateSRanges[0][1] - plateSRanges[0][0] + 1) 
					* (plateSRanges[1][1] - plateSRanges[1][0] + 1);
			plate[1] = (plate[0] + plateSize);

			// read the current version index into the shared memory 
			plateVersionIndex = plateBuffers.versionIndexBuffer[plateIndex];
		}
		__syncthreads();

		// The number of iterations to be performed on an LPU before the computation should move forward to the
		// next LPU is Space-A-padding / Space-B-padding
		for (int spaceAIter = 0; spaceAIter < (partition.padding1 / partition.padding2); spaceAIter++) {
			
			// before proceeding to the next iteration of an Space A LPU, all warps update on it's regions
			// must be synchronized with each other
                	__syncthreads();

			/*------------------------------------------------------------------------------------------------------------
                                                           		Space B Logic  
                        ------------------------------------------------------------------------------------------------------------*/
			
			// first determine the Space B LPU count for the current Space A LPU
			__shared__ int spaceBLpuCount1, spaceBLpuCount2, spaceBLpuCount;
			if (threadId == 0 && warpId == 0) {
				spaceBLpuCount1 = block_size_part_count(platePRanges[0], partition.blockSize);
				spaceBLpuCount2 = block_size_part_count(platePRanges[1], partition.blockSize);
				spaceBLpuCount = spaceBLpuCount1 * spaceBLpuCount2;
			}
			__syncthreads();

			// distribute the Space B LPUs among the warps
                        __shared__ int spaceBLpuId[WARP_COUNT][2];
			__shared__ int spaceBPRanges[WARP_COUNT][2][2];
			__shared__ int spaceBPRangesWithoutPadding[WARP_COUNT][2][2];
			__shared__ double *spaceBPlateParts[WARP_COUNT][2];
			__shared__ short warpEpochs[WARP_COUNT];
                        for (int spaceBLpu = warpId; spaceBLpu < spaceBLpuCount; spaceBLpu += WARP_COUNT) { 
				
				if (threadId == 0) {
					// construct the 2 dimensional LPU ID from the linear LPU Id
                                        spaceBLpuId[warpId][0] = spaceBLpu / spaceBLpuCount2;
                                        spaceBLpuId[warpId][1] = spaceBLpu % spaceBLpuCount2;
		
					// determine the region of the plate the current Space B LPU encompasses considering
					// padding; this range is the range over which the current LPU computation should 
					// take place	
					block_size_part_range(spaceBPRanges[warpId][0], platePRanges[0],
                                                        spaceBLpuCount1, spaceBLpuId[warpId][0], 
							partition.blockSize, 
							partition.padding2, partition.padding2);
                                        block_size_part_range(spaceBPRanges[warpId][1], platePRanges[1],
                                                        spaceBLpuCount2, spaceBLpuId[warpId][1],
                                                        partition.blockSize, 
							partition.padding2, partition.padding2);

					// determine another region information that excludes the padding; this range will be
					// used to resynchronize the Space B LPU update to the card memory Space A LPU region	
					block_size_part_range(spaceBPRangesWithoutPadding[warpId][0], platePRanges[0],
                                                        spaceBLpuCount1, spaceBLpuId[warpId][0], 
							partition.blockSize, 0, 0);
                                        block_size_part_range(spaceBPRangesWithoutPadding[warpId][1], platePRanges[1],
                                                        spaceBLpuCount2, spaceBLpuId[warpId][1],
							partition.blockSize, 0, 0);
				}
				// there is no syncthread operation needed here as updates done by a thread in a warp is 
				// visible to all other threads in that warp

				// before computation for an Space B LPU can begin, we need to allocate a separate memory
				// units for holding its plate parts as there are overlappings among Space B LPUs too and one 
				// warp computation on the padded region may corrupt other warps data
				if (threadId == 0) {
					spaceBPlateParts[warpId][0] = (double *) malloc(sizeof(double)
							* (spaceBPRanges[warpId][0][1] - spaceBPRanges[warpId][0][0] + 1)
							* (spaceBPRanges[warpId][1][1] - spaceBPRanges[warpId][1][0] + 1));	
					spaceBPlateParts[warpId][1] = (double *) malloc(sizeof(double)
							* (spaceBPRanges[warpId][0][1] - spaceBPRanges[warpId][0][0] + 1)
							* (spaceBPRanges[warpId][1][1] - spaceBPRanges[warpId][1][0] + 1));	
				}

				// all threads in the warp should collectively copy elements in the region that composes 
				// the current Space B LPU from the larger Space A LPU into the allocated memories 
				for (int i = spaceBPRanges[warpId][0][0]; i <= spaceBPRanges[warpId][0][1]; i++) {
					for (int j = spaceBPRanges[warpId][1][0] + threadId;
							j <= spaceBPRanges[warpId][1][1]; j += WARP_SIZE) {
						spaceBPlateParts[warpId][0][
								(i - spaceBPRanges[warpId][0][0]) 
								* (spaceBPRanges[warpId][1][1] 
										- spaceBPRanges[warpId][1][0] + 1)
								+ (j - spaceBPRanges[warpId][1][0])] 
							= plate[0][(i - plateSRanges[0][0]) 
								* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
								+ (j - plateSRanges[1][0])];  
						spaceBPlateParts[warpId][1][
								(i - spaceBPRanges[warpId][0][0]) 
								* (spaceBPRanges[warpId][1][1] 
										- spaceBPRanges[warpId][1][0] + 1)
								+ (j - spaceBPRanges[warpId][1][0])] 
							= plate[1][(i - plateSRanges[0][0]) 
								* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
								+ (j - plateSRanges[1][0])];  
					}
				}
				
				// set-up the epoch version of the Space B LPU matching the epoch version of the Space A
				// LPU it is a part of
				warpEpochs[warpId] = plateVersionIndex;

				// the computation on the warp's current Space B LPU will repeat up until the padding region 
				// has been depleted
				for (int spaceBIter = 0; spaceBIter < partition.padding2; spaceBIter++) {
					
					// first advance the Space B epoch version
					if (threadId == 0) {
						warpEpochs[warpId] = (warpEpochs[warpId] + 1) % 2;
					}
	
					// Distribute rows and columns of the plate cells to different threads. Note that
					// under the current scheme only rows or only columns will be distributed among the
					// threads of the warp, every thread will cover each entry in the other dimension.
					// Also notice that we have some arithmatic performed first before assigning indexes
					// to the threads. This is because the IT loop of the corresponding compute stage
					// has additional restrictions.
					int iterableRanges[4];
					iterableRanges[0] = spaceBPRanges[warpId][0][0] + 1;
					iterableRanges[1] = spaceBPRanges[warpId][0][1] - 1;
					iterableRanges[2] = spaceBPRanges[warpId][1][0] + 1;
					iterableRanges[3] = spaceBPRanges[warpId][1][1] - 1;
					int indexesAndSteps[4];
					determineLoopIndexesAndSteps(2, threadId, iterableRanges, indexesAndSteps);

					// iterate over the rows
					int iStart = indexesAndSteps[0];
					int iEnd = iterableRanges[1];
					int iStep = indexesAndSteps[1];
					for (int i = iStart; i <= iEnd; i += iStep) {

						// iterate over the columns
						int jStart = indexesAndSteps[2];
						int jEnd = iterableRanges[3];
						int jStep = indexesAndSteps[3];
						for (int j = jStart; j <= jEnd; j+= jStep) {
							// calculate the plate cell value for the current epoch as the average
							// of its four neighbors from the last epoch
							spaceBPlateParts[warpId][warpEpochs[warpId]]
									[(i - spaceBPRanges[warpId][0][0])
                                                                	* (spaceBPRanges[warpId][1][1] 
										- spaceBPRanges[warpId][1][0] + 1)
                                                                	+ (j - spaceBPRanges[warpId][1][0])]  
								= 0.25 * (spaceBPlateParts[warpId][(warpEpochs[warpId] + 1) % 2]
                                                                        	[(i + 1 - spaceBPRanges[warpId][0][0])
                                                                        	* (spaceBPRanges[warpId][1][1]
                                                                                	- spaceBPRanges[warpId][1][0] + 1)
                                                                        	+ (j - spaceBPRanges[warpId][1][0])]
									+ spaceBPlateParts[warpId][(warpEpochs[warpId] + 1) % 2]
                                                                                [(i - 1 - spaceBPRanges[warpId][0][0])
                                                                                * (spaceBPRanges[warpId][1][1]
                                                                                        - spaceBPRanges[warpId][1][0] + 1)
                                                                                + (j - spaceBPRanges[warpId][1][0])]
									+ spaceBPlateParts[warpId][(warpEpochs[warpId] + 1) % 2]
                                                                                [(i - spaceBPRanges[warpId][0][0])
                                                                                * (spaceBPRanges[warpId][1][1]
                                                                                        - spaceBPRanges[warpId][1][0] + 1)
                                                                                + (j + 1 - spaceBPRanges[warpId][1][0])]
									+ spaceBPlateParts[warpId][(warpEpochs[warpId] + 1) % 2]
                                                                                [(i - spaceBPRanges[warpId][0][0])
                                                                                * (spaceBPRanges[warpId][1][1]
                                                                                        - spaceBPRanges[warpId][1][0] + 1)
                                                                                + (j - 1 - spaceBPRanges[warpId][1][0])]);
						} // done iterating over j indices
					} // done iterating over i indices
				} // done Space B padding number of iterations
				
				// at the end of the computation, copy updates from the allocated memory for the Space B LPU
				// to the Space A LPU region 
				for (int i = spaceBPRangesWithoutPadding[warpId][0][0]; 
						i <= spaceBPRangesWithoutPadding[warpId][0][1]; i++) {
					for (int j = spaceBPRangesWithoutPadding[warpId][1][0] + threadId;
							j <= spaceBPRangesWithoutPadding[warpId][1][1]; j += WARP_SIZE) {
						plate[0][(i - plateSRanges[0][0]) 
								* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
								+ (j - plateSRanges[1][0])]  
							= spaceBPlateParts[warpId][0][
								(i - spaceBPRanges[warpId][0][0]) 
								* (spaceBPRanges[warpId][1][1] 
										- spaceBPRanges[warpId][1][0] + 1)
								+ (j - spaceBPRanges[warpId][1][0])]; 
						plate[1][(i - plateSRanges[0][0]) 
								* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
								+ (j - plateSRanges[1][0])]  
							= spaceBPlateParts[warpId][1][
								(i - spaceBPRanges[warpId][0][0]) 
								* (spaceBPRanges[warpId][1][1] 
										- spaceBPRanges[warpId][1][0] + 1)
								+ (j - spaceBPRanges[warpId][1][0])]; 
					}
				}
								
				// finally, the allocated memory should be freed
				if (threadId == 0) {
					free(spaceBPlateParts[warpId][0]);
					free(spaceBPlateParts[warpId][1]);
				}  
			}

			// How does a compiler decide to do the following? This is something that we probably cannot have in a
			// generated code and we needs to find an alternative. 
			__syncthreads();
			if (threadId == 0 && warpId == 0) {
				plateVersionIndex = (plateVersionIndex + 1) % 2;
			}
		}
	}
		
}

//------------------------------------------------------- Stencil GPU Code Executor ------------------------------------------------------/

StencilGpuCodeExecutor::StencilGpuCodeExecutor(LpuBatchController *lpuBatchController,
		stencil::Partition partition,
		stencil::ArrayMetadata arrayMetadata,
		stencil::TaskGlobals *taskGlobals,
		stencil::ThreadLocals *threadLocals) : GpuCodeExecutor(lpuBatchController) {

	this->partition = partition;
        this->arrayMetadata = arrayMetadata;
        this->taskGlobalsCpu = taskGlobals;
        this->taskGlobalsGpu = NULL;
        this->threadLocalsCpu = threadLocals;
        this->threadLocalsGpu = NULL;
}

void StencilGpuCodeExecutor::offloadFunction() {
	
	GpuBufferReferences *buffers = lpuBatchController->getGpuBufferReferences("plate");
	VersionedGpuBufferReferences *plateBuffers = (VersionedGpuBufferReferences *) buffers;
	
	StencilLpuBatchRange batchRange;
        batchRange.lpuIdRange = currentBatchLpuRange;
        batchRange.lpuCount = lpuCount[0];
	
	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
	stencilKernel1 <<< BLOCK_COUNT, threadsPerBlock >>> 
			(batchRange, partition, arrayMetadata, 
			taskGlobalsGpu, threadLocalsGpu, *plateBuffers);
	delete buffers;
}

void StencilGpuCodeExecutor::initialize() {

        GpuCodeExecutor::initialize();

        size_t taskGlobalsSize = sizeof(taskGlobalsCpu);
        cudaMalloc((void **) &taskGlobalsGpu, taskGlobalsSize);
        cudaMemcpy(taskGlobalsGpu, taskGlobalsCpu, taskGlobalsSize, cudaMemcpyHostToDevice);

        size_t threadLocalsSize = sizeof(threadLocalsCpu);
        cudaMalloc((void **) &threadLocalsGpu, threadLocalsSize);
        cudaMemcpy(threadLocalsGpu, threadLocalsCpu, threadLocalsSize, cudaMemcpyHostToDevice);
}

void StencilGpuCodeExecutor::cleanup() {

        size_t taskGlobalsSize = sizeof(taskGlobalsCpu);
        cudaMemcpy(taskGlobalsCpu, taskGlobalsGpu, taskGlobalsSize, cudaMemcpyDeviceToHost);
        size_t threadLocalsSize = sizeof(threadLocalsCpu);
        cudaMemcpy(threadLocalsCpu, threadLocalsGpu, threadLocalsSize, cudaMemcpyDeviceToHost);

        GpuCodeExecutor::cleanup();
}
