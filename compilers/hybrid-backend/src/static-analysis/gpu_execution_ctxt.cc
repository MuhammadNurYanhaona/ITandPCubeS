#include "gpu_execution_ctxt.h"
#include "data_flow.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>

//--------------------------------------------------------- GPU Execution Context ----------------------------------------------------------/

GpuExecutionContext::GpuExecutionContext(int topmostGpuPps, List<FlowStage*> *contextFlow) {
	this->contextFlow = contextFlow;
	Space *entryLps = contextFlow->Nth(0)->getSpace();
	this->contextLps = getContextLps(topmostGpuPps, entryLps);
	if (contextLps->getSubpartition() != NULL) {
		contextType = LOCATION_SENSITIVE_LPU_DISTR_CONTEXT;
	} else contextType = LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT;
}

int GpuExecutionContext::getContextId() {
	return contextFlow->Nth(0)->getIndex();
}

const char *GpuExecutionContext::getContextName() {
	std::ostringstream stream;
	int contextId = contextFlow->Nth(0)->getIndex();
	stream << "GpuExecutionContextNo" << contextId;
	return strdup(stream.str().c_str());
}

void GpuExecutionContext::describe(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << getContextName() << "\n";
	indent << '\t';
	std::cout << indent.str() << "Context LPS: " << contextLps->getName() << "\n";
	std::cout << indent.str() << "Flow Stages:" << "\n";
	indent << '\t';
	for (int i = 0; i < contextFlow->NumElements(); i++) {
		FlowStage *stage = contextFlow->Nth(i);
		std::cout << indent.str() << "Stage: " << stage->getName();
		std::cout << " (Space " << stage->getSpace()->getName() << ")\n";
	}
}

Space *GpuExecutionContext::getContextLps(int topmostGpuPps, Space *entryStageLps) {
	Space *candidateContextLps = entryStageLps;
	Space *currentLps = entryStageLps;
	while (!currentLps->isRoot() && currentLps->getPpsId() <= topmostGpuPps) {
		candidateContextLps = currentLps;
		currentLps = currentLps->getParent();
	}
	return candidateContextLps;
}


