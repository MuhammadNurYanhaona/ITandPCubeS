#include <cuda.h>
#include <cuda_runtime.h>

#include "mmm_gpu_execution.h"
#include "../../test-case/mmm_structure.h"
#include "../../runtime/structure.h"
#include "../../gpu-offloader/gpu_constant.h"
#include "../../gpu-offloader/gpu_code_executor.h"
#include "../../gpu-offloader/lpu_parts_tracking.h"
#include "../../utils/list.h"

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

