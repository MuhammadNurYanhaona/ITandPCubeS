#include <cuda.h>
#include <cuda_runtime.h>

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

	setBufferManager(new LpuDataBufferManager(versionlessProperties, multiversionProperties));
        initialize(lpuCountThreshold, memConsumptionLimit, propertyNames, toBeModifiedProperties);
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
