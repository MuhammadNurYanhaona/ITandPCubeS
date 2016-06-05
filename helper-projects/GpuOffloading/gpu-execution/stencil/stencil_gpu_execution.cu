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


// This kernel version assumes that Space A has been mapped to the SMs and Space B to the warps
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

// this kernel version assumes that Space A has been mapped to the GPU and Space B is mapped to the SMs
__global__ void stencilKernel2(StencilLpuBatchRange batchRange,
                stencil::Partition partition,
                stencil::ArrayMetadata arrayMetadata,
                stencil::TaskGlobals *taskGlobals,
                stencil::ThreadLocals *threadLocals,
                VersionedGpuBufferReferences plateBuffers, 
		int iteration) {

        // before we can do anything in the kernel, we need to determine the thread, warp, and sm IDs of the thread
        // executing the kernel code
        int smId = blockIdx.x;
        int warpId = threadIdx.x / WARP_SIZE;
        int threadId = threadIdx.x % WARP_SIZE;

        /*----------------------------------------------------------------------------------------------------------------------------
                                                                Space A Logic
        ----------------------------------------------------------------------------------------------------------------------------*/

        // variables holding the data part references for the current top level LPU SMs will be working on
        __shared__ double *plate[2];
        __shared__ int plateSRanges[2][2], platePRanges[2][2];

	// we are having trouble using dynamic shared memory; for now let us assume that the maximum Space B LPU size 
	// for any Space A LPU will not be larger than an arbitrary constant and we allocate that much shared memory 
	// space statically within the kernel
	__shared__ double plate_shared[2][2000];

	// an index to get a reference to the most up-to-date version of the place for an Space A LPU 
	__shared__ short plateVersionIndex;
	
	// all SMs go through all LPUs offloaded from the host
	Range lpuIdRange = batchRange.lpuIdRange;
        for (int lpuId = lpuIdRange.min; lpuId <= lpuIdRange.max; lpuId++) {

                // all threads should synchronize here to prevent the LPU metadata writer threads to overwrite the old
                // values before the other threads are done using those values
                __syncthreads(); 

		// the first thread in the SM reads LPU metadata information and initializes pointers 
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
                        plateVersionIndex = iteration % 2;
                }
                __syncthreads();

		/*--------------------------------------------------------------------------------------------------------------------
								Space B: Middle LPS
		--------------------------------------------------------------------------------------------------------------------*/ 

		// The first thread in each SM determines the Space B LPU count; this time we do not have the encircling
		// Space A padding1/padding2 number of iterations as those iterations will be done at the host level
		__shared__ int spaceBLpuCount1, spaceBLpuCount2, spaceBLpuCount;
		if (threadId == 0 && warpId == 0) {
			spaceBLpuCount1 = block_size_part_count(platePRanges[0], partition.blockSize);
			spaceBLpuCount2 = block_size_part_count(platePRanges[1], partition.blockSize);
			spaceBLpuCount = spaceBLpuCount1 * spaceBLpuCount2;
		}
		__syncthreads();

		// The Space B LPUs are distributed among different SMs/blocks. Although we are operating over multiple 
		// Space A LPUs in each kernel call, the deterministic nature of the LPU partition will ensure that a 
		// particular SM/block will always get the same set of Space B LPUs for offloaded parent Space A LPUs in 
		// all kernel invocations. It may be possible to utilize this fact to optimize the kernel executions. 
		// distribute the Space B LPUs among the warps
		__shared__ int spaceBLpuId[2];
		__shared__ int spaceBPRanges[2][2];
		__shared__ int spaceBPRangesWithoutPadding[2][2];
		__shared__ short spaceBVersionIndex;
                for (int spaceBLpu = smId; spaceBLpu < spaceBLpuCount; spaceBLpu += BLOCK_COUNT) {
		
			// a syncthread is needed in this step to stop the first thread in the SM from changing the part
			// dimension metadata for plate before all other threads have finished using that information	
			__syncthreads();

			if (threadId == 0 && warpId == 0) {
				// construct the 2 dimensional LPU ID from the linear LPU Id
				spaceBLpuId[0] = spaceBLpu / spaceBLpuCount2;
				spaceBLpuId[1] = spaceBLpu % spaceBLpuCount2;

				// determine the region of the plate the current Space B LPU encompasses considering
				// padding; this range is the range over which the current LPU computation should 
				// take place   
				block_size_part_range(spaceBPRanges[0], platePRanges[0],
						spaceBLpuCount1, spaceBLpuId[0],
						partition.blockSize,
						partition.padding2, partition.padding2);
				block_size_part_range(spaceBPRanges[1], platePRanges[1],
						spaceBLpuCount2, spaceBLpuId[1],
						partition.blockSize,
						partition.padding2, partition.padding2);

				// determine another region information that excludes the padding; this range will be
				// used to resynchronize the Space B LPU update to the card memory Space A LPU region   
				block_size_part_range(spaceBPRangesWithoutPadding[0], platePRanges[0],
						spaceBLpuCount1, spaceBLpuId[0],
						partition.blockSize, 0, 0);
				block_size_part_range(spaceBPRangesWithoutPadding[1], platePRanges[1],
						spaceBLpuCount2, spaceBLpuId[1],
						partition.blockSize, 0, 0);
				
				// set the version index of the space B LPU
				spaceBVersionIndex = plateVersionIndex;
			}
			__syncthreads();

			// all warps collectively read the current Space B plate portion from the global card memory to
			// the SM shared memory
			// different warps read different rows
			for (int i = spaceBPRanges[0][0] + warpId; i <= spaceBPRanges[0][1]; i += WARP_COUNT) {
				// different threads read different columns
				for (int j = spaceBPRanges[1][0] + threadId;
						j <= spaceBPRanges[1][1]; j += WARP_SIZE) {
					plate_shared[0][(i - spaceBPRanges[0][0])
							* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
							+ (j - spaceBPRanges[1][0])]
						= plate[0][(i - plateSRanges[0][0])
							* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
							+ (j - plateSRanges[1][0])];
					plate_shared[1][(i - spaceBPRanges[0][0])
							* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
							+ (j - spaceBPRanges[1][0])]
						= plate[1][(i - plateSRanges[0][0])
							* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
							+ (j - plateSRanges[1][0])];
				}
			}
			__syncthreads();

			for (int spaceBIter = 0; spaceBIter < partition.padding2; spaceBIter++) {

				// first advance the Space B epoch version
				if (threadId == 0 && warpId == 0) {
					spaceBVersionIndex = (spaceBVersionIndex + 1) % 2;
				}
				__syncthreads();

				// just like in the matrix-matrix multiplication, we assume that the partition hierarchy
				// in this scenaro has another lower level LPS that distributes regions of the Space B
				// LPU to different warps
				/*----------------------------------------------------------------------------------------------------
									 Space C: Bottom LPS
				----------------------------------------------------------------------------------------------------*/ 
				__shared__ int spaceCLpuCount1, spaceCLpuCount2, spaceCLpuCount;
				if (threadId == 0 && warpId == 0) {
					spaceCLpuCount1 = block_size_part_count(spaceBPRanges[0], 1);
					spaceCLpuCount2 = block_size_part_count(spaceBPRanges[1], partition.blockSize);
					spaceCLpuCount = spaceCLpuCount1 * spaceCLpuCount2;
				}
				__syncthreads();

				// distribute the Space C LPUs among the warps
                        	__shared__ int spaceCLpuId[WARP_COUNT][2];
                        	__shared__ int spaceCPRanges[WARP_COUNT][2][2];
				for (int spaceCLpu = warpId; spaceCLpu < spaceCLpuCount; spaceCLpu += WARP_COUNT) {

					if (threadId == 0) {
						// construct the 2 dimensional LPU ID from the linear LPU Id
						spaceCLpuId[warpId][0] = spaceCLpu / spaceCLpuCount2;
						spaceCLpuId[warpId][1] = spaceCLpu % spaceCLpuCount2;

						// determine the region of the plate the current Space C LPU encompasses
						block_size_part_range(spaceCPRanges[warpId][0], spaceBPRanges[0],
								spaceCLpuCount1, spaceCLpuId[warpId][0],
								1, 1, 1);
						block_size_part_range(spaceCPRanges[warpId][1], spaceBPRanges[1],
								spaceCLpuCount2, spaceCLpuId[warpId][1],
								partition.blockSize, 1, 1);
					}
					// there is no syncthread operation needed here as updates done by a thread in a warp is 
					// visible to all other threads in that warp
					
					// Distribute rows and columns of the plate cells to different threads. Note that
					// under the current scheme only rows or only columns will be distributed among the
					// threads of the warp, every thread will cover each entry in the other dimension.
					// Also notice that we have some arithmatic performed first before assigning indexes
					// to the threads. This is because the IT loop of the corresponding compute stage
					// has additional restrictions.
					int iterableRanges[4];
					iterableRanges[0] = spaceCPRanges[warpId][0][0] + 1;
					iterableRanges[1] = spaceCPRanges[warpId][0][1] - 1;
					iterableRanges[2] = spaceCPRanges[warpId][1][0] + 1;
					iterableRanges[3] = spaceCPRanges[warpId][1][1] - 1;
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
							plate_shared[spaceBVersionIndex]
									[(i - spaceBPRanges[0][0])
                                                                	* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
                                                                	+ (j - spaceBPRanges[1][0])]  
								= 0.25 * (plate_shared[(spaceBVersionIndex + 1) % 2]
                                                                        	[(i + 1 - spaceBPRanges[0][0])
                                                                        	* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
                                                                        	+ (j - spaceBPRanges[1][0])]
									+ plate_shared[(spaceBVersionIndex + 1) % 2]
                                                                                [(i - 1 - spaceBPRanges[0][0])
                                                                                * (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
                                                                                + (j - spaceBPRanges[1][0])]
									+ plate_shared[(spaceBVersionIndex + 1) % 2]
                                                                                [(i - spaceBPRanges[0][0])
                                                                                * (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
                                                                                + (j + 1 - spaceBPRanges[1][0])]
									+ plate_shared[(spaceBVersionIndex + 1) % 2]
                                                                                [(i - spaceBPRanges[0][0])
                                                                                * (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
                                                                                + (j - 1 - spaceBPRanges[1][0])]);
						} // done iterating over j indices
					} // done iterating over i indices
				} // done iterating over Space C LPUs	
			} // done doing padding2 number of Space B iterations 
			
						
			// finally the update done by the SM is written back to the correct version of the Space A LPU
			// in the global memory; now this cannot be done in the general case as the update will skew the
			// view of the other SM's padding region data since they are expecting to read old cell values
			// as opposed to the updated values. We are doing this here because we know that the new version
			// wont be used before the next kernel launch. A lot of static analysis in the source code will
			// be needed to determine that this kind of updates will not corrupt data. 
			// different warps read different rows
			for (int i = spaceBPRangesWithoutPadding[0][0] + warpId; 
					i <= spaceBPRangesWithoutPadding[0][1]; i += WARP_COUNT) {
				// different threads read different columns
				for (int j = spaceBPRangesWithoutPadding[1][0] + threadId;
						j <= spaceBPRangesWithoutPadding[1][1]; j += WARP_SIZE) {
					plate[spaceBVersionIndex][(i - plateSRanges[0][0])
							* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
							+ (j - plateSRanges[1][0])]
						= plate_shared[spaceBVersionIndex][(i - spaceBPRanges[0][0])
							* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
							+ (j - spaceBPRanges[1][0])];
					plate[(spaceBVersionIndex + 1) % 2][(i - plateSRanges[0][0])
							* (plateSRanges[1][1] - plateSRanges[1][0] + 1)
							+ (j - plateSRanges[1][0])]
						= plate_shared[(spaceBVersionIndex + 1) % 2][(i - spaceBPRanges[0][0])
							* (spaceBPRanges[1][1] - spaceBPRanges[1][0] + 1)
							+ (j - spaceBPRanges[1][0])];
				} // iteration over i ends
			} // iteration over j ends
		} // iterating over Space B LPUs ends
	} // iterating over Space A LPUs ends
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

void StencilGpuCodeExecutor::offloadFunction1() {

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

void StencilGpuCodeExecutor::offloadFunction2() {

	GpuBufferReferences *buffers = lpuBatchController->getGpuBufferReferences("plate");
        VersionedGpuBufferReferences *plateBuffers = (VersionedGpuBufferReferences *) buffers;

        StencilLpuBatchRange batchRange;
        batchRange.lpuIdRange = currentBatchLpuRange;
        batchRange.lpuCount = lpuCount[0];
        int threadsPerBlock = WARP_SIZE * WARP_COUNT;

	int spaceAIterations = partition.padding1 / partition.padding2;

	for (int i = 0; i < spaceAIterations; i++) {
        	stencilKernel2 <<< BLOCK_COUNT, threadsPerBlock >>>
                        	(batchRange, partition, arrayMetadata,
                        	taskGlobalsGpu, threadLocalsGpu, *plateBuffers, 
				i);
	}

        delete buffers;
}

void StencilGpuCodeExecutor::offloadFunction() {
	offloadFunction2();	
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
