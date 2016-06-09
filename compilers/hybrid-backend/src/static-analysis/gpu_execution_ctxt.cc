#include "gpu_execution_ctxt.h"
#include "data_flow.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"

//--------------------------------------------------------- GPU Execution Context ----------------------------------------------------------/

GpuExecutionContext::GpuExecutionContext(Space *contextLps, List<FlowStage*> *contextFlow) {
	this->contextLps = contextLps;
	this->contextFlow = contextFlow;
	if (contextLps->getSubpartition() != NULL) {
		contextType = LOCATION_SENSITIVE_LPU_DISTR_CONTEXT;
	} else contextType = LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT;
}


