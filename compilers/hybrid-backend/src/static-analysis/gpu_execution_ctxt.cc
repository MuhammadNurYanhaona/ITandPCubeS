#include "gpu_execution_ctxt.h"
#include "data_flow.h"
#include "../syntax/ast_stmt.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <deque>

//--------------------------------------------------------- GPU Execution Context ----------------------------------------------------------/

Hashtable<GpuExecutionContext*> *GpuExecutionContext::gpuContextMap = NULL;

GpuExecutionContext::GpuExecutionContext(int topmostGpuPps, List<FlowStage*> *contextFlow) {
	this->contextFlow = contextFlow;
	Space *entryLps = contextFlow->Nth(0)->getSpace();
	this->contextLps = getContextLps(topmostGpuPps, entryLps);
	if (contextLps->getSubpartition() != NULL) {
		contextType = LOCATION_SENSITIVE_LPU_DISTR_CONTEXT;
	} else contextType = LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT;
	performVariableAccessAnalysis();
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

const char *GpuExecutionContext::generateContextName(int contextId) {
	std::ostringstream stream;
        stream << "GpuExecutionContextNo" << contextId;
        return strdup(stream.str().c_str());
}

List<const char*> *GpuExecutionContext::getVariableAccessList() {
	
	List<const char*> *varList = new List<const char*>;
	Iterator<VariableAccess*> iterator = varAccessLog->GetIterator();
	VariableAccess *varAccess = NULL;
	while ((varAccess = iterator.GetNextValue()) != NULL) {
		varList->Append(varAccess->getName());
	}
	return varList;
}
        
List<const char*> *GpuExecutionContext::getModifiedVariableList() {
	
	List<const char*> *varList = new List<const char*>;
	Iterator<VariableAccess*> iterator = varAccessLog->GetIterator();
	VariableAccess *varAccess = NULL;
	while ((varAccess = iterator.GetNextValue()) != NULL) {
		if (varAccess->isContentAccessed()) {	
			AccessFlags *accessFlags = varAccess->getContentAccessFlags();
			if (accessFlags->isWritten() || accessFlags->isRedirected()) {
				varList->Append(varAccess->getName());
			}
		}
	}
	return varList;
}
        
List<const char*> *GpuExecutionContext::getEpochIndependentVariableList() {
	List<const char*> *allVarAccesses = getVariableAccessList();
	return string_utils::subtractList(allVarAccesses, epochDependentVarAccesses);
}

void GpuExecutionContext::generateInvocationCode(std::ofstream &stream, int indentation, Space *callingCtxtLps) {

	// start an offloading scope, and retrieve and initialize the gpu code executor for this context
	std::ostringstream indent;
        for (int i = 0; i < indentation; i++) indent << '\t';
	stream << std::endl;
	stream << indent.str() << "{ // GPU LPU offloading context starts\n";
	stream << indent.str() << "GpuCodeExecutor *gpuCodeExecutor = batchPpuState->";
	stream << "getGpuExecutorForContext(\"" << getContextName() << "\")" << stmtSeparator;
	stream << indent.str() << "gpuCodeExecutor->initialize()" << stmtSeparator;
	 
	int transitionLpsCount = 0;
	List<Space*> *transitLpsList = new List<Space*>;
	Space *currentLps = contextLps->getParent();
	while (currentLps != callingCtxtLps) {
		transitionLpsCount++;
		transitLpsList->InsertAt(currentLps, 0);
		currentLps = currentLps->getParent();
	}

	int offloadingIndent = indentation + transitionLpsCount;
	const char *offloadingCode = spewOffloadingContextCode(offloadingIndent);
	
	// note that calling context LPS is not part of the transition LPSes but we insert it in the list 
	// nonetheless to initiate the code generation; the next to last paramater to the wrapping function is
	// starting at index 1 to skip the calling context LPS 
	transitLpsList->InsertAt(callingCtxtLps, 0);
	wrapOffloadingCodeInLargerContext(stream, indentation, transitLpsList, 1, offloadingCode);	

	// tear down the gpu offloading context and close the scope
	stream << '\n' << indent.str() << "gpuCodeExecutor->cleanup()" << stmtSeparator;
	stream << indent.str() << "} // GPU LPU offloading context ends\n";
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

void GpuExecutionContext::performVariableAccessAnalysis() {

	varAccessLog = new Hashtable<VariableAccess*>;
	epochDependentVarAccesses = new List<const char*>;

	std::deque<FlowStage*> stageQueue;
	for (int i = 0; i < contextFlow->NumElements(); i++) {
		stageQueue.push_back(contextFlow->Nth(i));
	}

	while (!stageQueue.empty()) {
		FlowStage *stage = stageQueue.front();
		stageQueue.pop_front();
		CompositeStage *metaStage = dynamic_cast<CompositeStage*>(stage);
		if (metaStage != NULL) {
			List<FlowStage*> *stageList = metaStage->getStageList();
			for (int i = 0; i < stageList->NumElements(); i++) {
				stageQueue.push_back(stageList->Nth(i));
			}
		}
		SyncStage *syncStage = dynamic_cast<SyncStage*>(stage);
		if (syncStage != NULL) continue;

		Hashtable<VariableAccess*> *stageAccesses = stage->getAccessMap();
		if (stageAccesses != NULL) {
			Stmt::mergeAccessedVariables(varAccessLog, stage->getAccessMap());
		}
		List<const char*> *stageEpochVars = stage->getEpochDependentVarList();
		if (stageEpochVars != NULL) {
			string_utils::combineLists(epochDependentVarAccesses, stageEpochVars);
		}
	}	
}

const char *GpuExecutionContext::spewOffloadingContextCode(int indentation) {

	std::ostringstream stream;
	
	std::ostringstream indent;
        for (int i = 0; i < indentation; i++) indent << '\t';
	
	const char *lpsName = contextLps->getName();
	Space *containerLps = contextLps->getParent();
	const char *containerLpsName = containerLps->getName();

	// create a new local scope for retrieving LPUs of the context LPS
	stream << std::endl;
	stream << indent.str() << "{ // scope entrance for gathering LPUs for offload\n";

	// declare a new ID vector to track progress in LPU generation and initialize it
	stream << indent.str() << "std::vector<int> lpuIdVector" << stmtSeparator;
	stream << indent.str() << "batchPpuState->initLpuIdVectorsForLPSTraversal(Space_";
	stream << lpsName << paramSeparator << "&lpuIdVector)" << stmtSeparator;

	// declare an initialize an iteration counter
	stream << indent.str() << "int iterationNo = 0" << stmtSeparator;

	// declare another vector to hold on to current LPUs of this LPS
	stream << indent.str() << "std::vector<LPU*> *lpuVector" << stmtSeparator;
                
	// generate LPUs by repeatedly invoking the get-next-LPU routine
	stream << indent.str() << "while((lpuVector = batchPpuState->getNextLpus(";
	stream << "Space_" << lpsName << paramSeparator << "Space_" << containerLpsName;
	stream << paramSeparator << "&lpuIdVector)) != NULL) {\n";
	
	// in the first iteration set up the LPU count in the GPU code executor
	stream << indent.str() << "\tif(iterationNo == 0) {\n";
	stream << indent.str() << doubleIndent;
	stream << "gpuCodeExecutor->setLpuCount(threadState->getLpuCounts(";
	stream << "Space_" << lpsName << "))" << stmtSeparator;
	stream << indent.str() << "\t}\n";

	// hand over the vector of LPUs to the GPU code executor
	stream << indent.str() << '\t' << "gpuCodeExecutor->submitNextLpus(lpuVector);\n";
	
	// update the LPU ID vector and iteration counter, and close the LPS traveral loop
	stream << indent.str() << '\t' << "batchPpuState->extractLpuIdsFromLpuVector(";
        stream << "&lpuIdVector" << paramSeparator << "lpuVector)" << stmtSeparator;
	stream << indent.str() << '\t' << "iterationNo++" << stmtSeparator;
       	stream << indent.str() << "}\n";
	
	// force execution of last remaining LPUs that did not fill up a complete batch
	stream << indent.str() << "gpuCodeExecutor->forceExecution()" << stmtSeparator;

	// exit from the scope
	stream << indent.str() << "} // scope exit for gathering LPUs for offload\n";

	return strdup(stream.str().c_str());
}

void GpuExecutionContext::wrapOffloadingCodeInLargerContext(std::ofstream &stream, int indentation,
		List<Space*> *transitLpsList, 
		int index, const char *offloadingCode) {

	if (index == transitLpsList->NumElements()) {
		stream << offloadingCode;
		return;
	} 

	std::ostringstream indent;
        for (int i = 0; i < indentation; i++) indent << '\t';

	Space *lps = transitLpsList->Nth(index);
	const char *lpsName = lps->getName();
	Space *containerLps = transitLpsList->Nth(index - 1);
	const char *containerLpsName = containerLps->getName();

	// create a new local scope for traversing LPUs of this new scope
	stream << std::endl;
	stream << indent.str() << "{ // scope entrance for iterating LPUs of Space " << lpsName << "\n";
                
	// declare a new ID vector to track progress in LPU generation and initialize it
	stream << indent.str() << "std::vector<int> lpuIdVector" << stmtSeparator;
	stream << indent.str() << "batchPpuState->initLpuIdVectorsForLPSTraversal(Space_";
	stream << lpsName << paramSeparator << "&lpuIdVector)" << stmtSeparator;

	// declare another vector to hold on to current LPUs of this LPS
	stream << indent.str() << "std::vector<LPU*> *lpuVector" << stmtSeparator;
                
	// generate LPUs by repeatedly invoking the get-next-LPU routine
	stream << indent.str() << "while((lpuVector = batchPpuState->getNextLpus(";
	stream << "Space_" << lpsName << paramSeparator << "Space_" << containerLpsName;
	stream << paramSeparator << "&lpuIdVector)) != NULL) {\n";

	// recursively go to the next transit LPS
	wrapOffloadingCodeInLargerContext(stream, indentation + 1, transitLpsList, index + 1, offloadingCode);

	// update the LPU ID vector and close the LPS traveral loop
	stream << indent.str() << '\t' << "batchPpuState->extractLpuIdsFromLpuVector(";
        stream << "&lpuIdVector" << paramSeparator << "lpuVector)" << stmtSeparator;
       	stream << indent.str() << "}\n";
                
	// at the end remove checkpoint if the container LPS is not the root LPS
	if (!containerLps->isRoot()) {
		stream << indent.str() << "batchPpuState->removeIterationBound(Space_";
		stream << containerLpsName << ')' << stmtSeparator;
	}

	// exit from the scope
	stream << indent.str() << "} // scope exit for iterating LPUs of Space " << lpsName << "\n";
}


