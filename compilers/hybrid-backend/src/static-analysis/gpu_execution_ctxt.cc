#include "gpu_execution_ctxt.h"
#include "data_flow.h"
#include "data_access.h"
#include "sync_stat.h"
#include "../syntax/ast_stmt.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"
#include "../codegen/space_mapping.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <deque>

//------------------------------------------------------- Kernel Group Configuration -------------------------------------------------------/

KernelGroupConfig::KernelGroupConfig(int groupId, List<FlowStage*> *contextSubflow) {
	this->groupId = groupId;
	this->repeatingKernels = false;
	this->repeatCondition = NULL;
	this->contextSubflow = contextSubflow;
	this->kernelConfigs = NULL;
}

KernelGroupConfig::KernelGroupConfig(int groupId, RepeatCycle *repeatCycle) {
	this->groupId = groupId;
	this->repeatingKernels = true;
	this->repeatCondition = repeatCycle->getRepeatCondition();
	this->contextSubflow = repeatCycle->getStageList();
	this->kernelConfigs = NULL;
}

void KernelGroupConfig::describe(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << "Kernel Config Group ID: " << groupId << "\n";
	std::cout << indent.str() << "Does Repeat? ";
	if (repeatingKernels) {
		std::cout << "Yes\n";
	} else {
		std::cout << "No\n";
	}
	std::cout << indent.str() << "Original Context Flow:\n";
	for (int i = 0; i < contextSubflow->NumElements(); i++) {
		contextSubflow->Nth(i)->print(indentLevel + 1);
	}
	if (kernelConfigs != NULL) {
		std::cout << indent.str() << "Translated GPU Execution Flow:\n";
		for (int i = 0; i < kernelConfigs->NumElements(); i++) {
			kernelConfigs->Nth(i)->print(indentLevel + 1);
		}
	}
}

void KernelGroupConfig::generateKernelConfig(PCubeSModel *pcubesModel, Space *contextLps) {
	
	std::deque<FlowStage*> stageQueue;
	for (int i = 0; i < contextSubflow->NumElements(); i++) {
		stageQueue.push_back(contextSubflow->Nth(i));
	}
	int gpuTransitionLevel = pcubesModel->getGpuTransitionSpaceId();
	CompositeStage *configUnderConstruct = new CompositeStage(0, contextLps, NULL);
	List<SyncRequirement*> *configSyncSignals = new List<SyncRequirement*>;

	// the empty top-level composite stage needs to be entered in the configuration list first as the recursive
	// helper routine expand its and move to deeper and deeper nesting level, the original stage reference gets
	// lost in the process
	List<CompositeStage*> *configList = new List<CompositeStage*>;
	configList->Append(configUnderConstruct);


	generateKernelConfig(&stageQueue, gpuTransitionLevel, 
			contextLps, configList, configUnderConstruct, configSyncSignals);
	kernelConfigs = configList;
}

void KernelGroupConfig::generateKernelConfig(std::deque<FlowStage*> *stageQueue,
		int gpuTransitionLevel,
		Space *contextLps,
		List<CompositeStage*> *currentConfigList,
		CompositeStage *configUnderConstruct, 
		List<SyncRequirement*> *configSyncSignals) {

	int smLevel = gpuTransitionLevel - 1;
	FlowStage *currentStage = stageQueue->front();
	stageQueue->pop_front();

	CompositeStage *nextConfigUnderConstr = configUnderConstruct;

	// check if the current stage have any synchronization dependency originating from the under construction
	// kernel that has communication/sync-root (check data_access.h for their definitions) above the SM level
	bool needKernelExit = false;
	for (int i = 0; i < configSyncSignals->NumElements(); i++) {
		DependencyArc *arc = configSyncSignals->Nth(i)->getDependencyArc();
		if (arc->getSignalSink() == currentStage) {
			Space *commRootLps = arc->getCommRoot();
			Space *syncRootLps = arc->getSyncRoot();
			if ((commRootLps != NULL && commRootLps->getPpsId() > smLevel) 
					|| (syncRootLps != NULL && syncRootLps->getPpsId() > smLevel)) {
				needKernelExit = true;
				break;
			}
		}
	}
	
	// if a kernel exit is needed at this position then create a new blank composite stage for the next kernel
	if (needKernelExit) {
		int stageId = currentConfigList->NumElements();
		CompositeStage *newKernelConfig = new CompositeStage(stageId, contextLps, NULL);
		// put in the root stage of the new kernel configuration in the list 
		currentConfigList->Append(newKernelConfig);
		// reset the sync requirements
		configSyncSignals->clear();
		StageSyncReqs *stageSyncs = currentStage->getAllSyncRequirements();
		List<SyncRequirement*> *syncSignals = stageSyncs->getAllSyncRequirements();
		configSyncSignals->AppendAll(syncSignals);
		// change the under construction configuration reference
		nextConfigUnderConstr = newKernelConfig;		
	} else {
		// add the sync signals of the current stage to the existing list of sync signals
		StageSyncReqs *stageSyncs = currentStage->getAllSyncRequirements();
		List<SyncRequirement*> *syncSignals = stageSyncs->getAllSyncRequirements();
		configSyncSignals->AppendAll(syncSignals);
	}

	// determine what kind of flow stage the current stage is
	SyncStage *syncStage = dynamic_cast<SyncStage*>(currentStage);
	CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(currentStage);
	ExecutionStage *executionStage = dynamic_cast<ExecutionStage*>(currentStage);  	  
	

	// If the current flow stage is a sync stage then we do not need to add that in the current kernel config.
	// we just need to track down its dependencies properly, which we did already
	if (syncStage != NULL) {
		// do nothing

	// if the current stage is an execution stage then we add that to the stage list of the current composite
	// stage under construction
	} else if (executionStage != NULL) {
		nextConfigUnderConstr->addStageAtEnd(executionStage);
	
	
	// If the current stage is a composite stage then we might need to probe its content to decide how further
	// kernel boundaries must be drawn, or we can just add the whole stage in the current kernel config. The
	// decision depends on the LPS the composite stage is executing on. If the LPS is at or below the SM level
	// then we do not need to probe further. If it is at the GPU level then we do.	
	} else if (compositeStage != NULL) {
		int compositeStageLevel = compositeStage->getSpace()->getPpsId();
		if (compositeStageLevel <= smLevel) {
			nextConfigUnderConstr->addStageAtEnd(compositeStage);
		} else {
			// add the content of the composite stage to the front of the stage queue
			 List<FlowStage*> *stageList = compositeStage->getStageList();
			for (int i = stageList->NumElements() - 1; i >= 0; i--) {
				stageQueue->push_front(stageList->Nth(i));
			}
		}
	}

	// finally, if there are more elements in the stage queue then invoke the same routine recursively
	if (!stageQueue->empty()) {
		generateKernelConfig(stageQueue, gpuTransitionLevel,
                        	contextLps, currentConfigList, nextConfigUnderConstr, configSyncSignals);
	}
}

//--------------------------------------------------------- GPU Execution Context ----------------------------------------------------------/

Hashtable<GpuExecutionContext*> *GpuExecutionContext::gpuContextMap = NULL;

GpuExecutionContext::GpuExecutionContext(int topmostGpuPps, List<FlowStage*> *contextFlow) {
	this->contextFlow = contextFlow;
	Space *entryLps = contextFlow->Nth(0)->getSpace();
	this->contextLps = getContextLps(topmostGpuPps, entryLps);
	if (contextLps->isSubpartitionSpace()) {
		contextType = LOCATION_SENSITIVE_LPU_DISTR_CONTEXT;
	} else contextType = LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT;
	performVariableAccessAnalysis();
	kernelConfigList = NULL;
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

void GpuExecutionContext::generateKernelConfigs(PCubeSModel *pcubesModel) {
	
	int gpuLevel = pcubesModel->getGpuTransitionSpaceId();

	// first peel off any subpartition repeat at the GPU context LPS as those repeat loops are automatically
	// handled in the LPU generation process 
	std::deque<FlowStage*> stageQueue;
	for (int i = 0; i < contextFlow->NumElements(); i++) {
		FlowStage *stage = contextFlow->Nth(i);
		RepeatCycle *repeatCycle = dynamic_cast<RepeatCycle*>(stage);
		if (repeatCycle != NULL && repeatCycle->isSubpartitionRepeat()) {
			List<FlowStage*> *stageList = repeatCycle->getStageList();
			for (int j = 0; j < stageList->NumElements(); j++) {
				stageQueue.push_back(stageList->Nth(j));
			}
			continue;
		}
		stageQueue.push_back(stage); 
	}

	kernelConfigList = new List<KernelGroupConfig*>; 
	
	// If the current GPU context LPS itself has been mapped to the SM or Warp level then there will be just one
	// kernel group and no host level repetition is needed even for the outermost repeat loops as the subflow
	// can be executed in the GPU entirely.
	if (contextLps->getPpsId() < gpuLevel) {
		List<FlowStage*> *kernelGroupFlow = new List<FlowStage*>;
		while (!stageQueue.empty()) {
			kernelGroupFlow->Append(stageQueue.front());
			stageQueue.pop_front();
		}
		KernelGroupConfig *groupConfig = new KernelGroupConfig(0, kernelGroupFlow);
		groupConfig->generateKernelConfig(pcubesModel, contextLps);
		kernelConfigList->Append(groupConfig);

	// Otherwise, distinction needs to be made between the parts of the context flow that will repeat from those
	// that will not as the top level repeat block will execute in the host	
	} else {
		int groupIndex = 0;
		List<FlowStage*> *currentSubflow = new List<FlowStage*>;
		while (!stageQueue.empty()) {
			FlowStage *stage = stageQueue.front();
			stageQueue.pop_front();
			RepeatCycle *repeatCycle = dynamic_cast<RepeatCycle*>(stage);
			if (repeatCycle != NULL) {
				if (currentSubflow->NumElements() > 0) {
					KernelGroupConfig *groupConfig = new KernelGroupConfig(groupIndex, 
						currentSubflow);
					groupConfig->generateKernelConfig(pcubesModel, contextLps);
					kernelConfigList->Append(groupConfig);
					groupIndex++;
				}
				KernelGroupConfig *repeatConfig = new KernelGroupConfig(groupIndex, repeatCycle);
				repeatConfig->generateKernelConfig(pcubesModel, contextLps);
				kernelConfigList->Append(repeatConfig);	
				groupIndex++;
			}
		}
		if (currentSubflow->NumElements() > 0) {
			KernelGroupConfig *groupConfig = new KernelGroupConfig(groupIndex, 
				currentSubflow);
			groupConfig->generateKernelConfig(pcubesModel, contextLps);
			kernelConfigList->Append(groupConfig);
		}
	}
}

void GpuExecutionContext::generateInvocationCode(std::ofstream &stream, 
		int indentation, Space *callingCtxtLps) {

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
	for (int i = 0; i < contextFlow->NumElements(); i++) {
		contextFlow->Nth(i)->print(indentLevel + 2);
	}
	if (kernelConfigList != NULL) {
		std::cout << indent.str() << "Kernel Group Configurations:" << "\n";
		for (int i = 0; i < kernelConfigList->NumElements(); i++) {
			kernelConfigList->Nth(i)->describe(indentLevel + 2);
		}
	}
}

Space *GpuExecutionContext::getContextLps(int topmostGpuPps, Space *entryStageLps) {
	Space *candidateContextLps = entryStageLps;
	Space *currentLps = entryStageLps;
	while (!currentLps->isRoot() && currentLps->getPpsId() <= topmostGpuPps) {
		candidateContextLps = currentLps;
		currentLps = currentLps->getParent();
	}
	if (candidateContextLps->getSubpartition() != NULL) {
		return candidateContextLps->getSubpartition();
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
	stream << indent.str() << "std::vector<int> contextLpuIdVector" << stmtSeparator;
	stream << indent.str() << "batchPpuState->initLpuIdVectorsForLPSTraversal(Space_";
	stream << lpsName << paramSeparator << "&contextLpuIdVector)" << stmtSeparator;

	// if the current GPU context requires PPU location sensitive LPU distribution then
	// further adjust the LPU ID vector based on the presence of a non-NULL LPU in the
	// ancestor for individual PPU controllers
	if (contextType == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {
		stream << indent.str() << "batchPpuState->adjustLpuIdVector(Space_";
		stream << lpsName << paramSeparator << paramIndent << indent.str(); 
		stream << "&contextLpuIdVector" << paramSeparator;
		stream << "Space_" << containerLpsName << paramSeparator;
		stream << "lpuVector)" << stmtSeparator;
	}

	// declare an initialize an iteration counter
	stream << indent.str() << "int iterationNo = 0" << stmtSeparator;

	// declare another vector to hold on to current LPUs of this LPS
	stream << indent.str() << "std::vector<LPU*> *contextLpuVector" << stmtSeparator;
                
	// generate LPUs by repeatedly invoking the get-next-LPU routine
	stream << indent.str() << "while((contextLpuVector = batchPpuState->getNextLpus(";
	stream << "Space_" << lpsName << paramSeparator << "Space_" << containerLpsName;
	stream << paramSeparator << "&contextLpuIdVector)) != NULL) {\n";
	
	// in the first iteration set up the LPU count in the GPU code executor
	stream << indent.str() << "\tif(iterationNo == 0) {\n";
	stream << indent.str() << doubleIndent;
	stream << "gpuCodeExecutor->setLpuCountVector(batchPpuState->genLpuCountsVector(";
	stream << "Space_" << lpsName << paramSeparator; 
	if (contextType == LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT) {
		stream << "true";
	} else { 
		stream << "false";
	}
	stream << "))" << stmtSeparator;
	stream << indent.str() << "\t}\n";

	// hand over the vector of LPUs to the GPU code executor
	stream << indent.str() << '\t';
	stream << "gpuCodeExecutor->submitNextLpus(contextLpuVector);\n";
	
	// update the LPU ID vector and iteration counter, and close the LPS traveral loop
	stream << indent.str() << '\t' << "batchPpuState->extractLpuIdsFromLpuVector(";
        stream << "&contextLpuIdVector" << paramSeparator;
	stream << "contextLpuVector)" << stmtSeparator;
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


