#include "gpu_execution_ctxt.h"
#include "data_flow.h"
#include "data_access.h"
#include "sync_stat.h"
#include "../syntax/ast_stmt.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/code_constant.h"
#include "../utils/string_utils.h"
#include "../utils/decorator_utils.h"
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
	
	int topmostGpuPpsForContext = contextLps->getPpsId();
	kernelConfigs = new List<CompositeStage*>;
	for (int i = 0; i < configList->NumElements(); i++) {
		CompositeStage *prospectiveKernel = configList->Nth(i);
		if (prospectiveKernel->isEmpty()) continue;
		
		// we make all LPS transitions inside the kernel to be non-LPS jumping (means the transition skips
		// no intermediate LPS) to be able to do intra-kernel LPS expansition easily
		prospectiveKernel->makeAllLpsTransitionExplicit();

		prospectiveKernel->setupGpuLpuDistrFlags(topmostGpuPpsForContext);
		kernelConfigs->Append(prospectiveKernel);
	}
}

void KernelGroupConfig::generateKernelGroupExecutionCode(std::ofstream &programFile, 
			List<const char*> *accessedArrays, int indentLevel) {
	
	if (repeatingKernels) {
		std::cout << "We still haven't implement the logic for repititive kernels. Please wait. :P\n";
		std::exit(EXIT_FAILURE);
	}

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;

	for (int i = 0; i < kernelConfigs->NumElements(); i++) {
		
		CompositeStage *kernelDef = kernelConfigs->Nth(i);
		const char *kernelName = kernelDef->getName();
		
		//---------------------------------------------------------------------------kernel invocation
		
		programFile << indentStr.str() << kernelName;
		
		// kernel launch config
		programFile << paramIndent << indentStr.str();
		programFile << " <<< gridConfig" << paramSeparator;
		programFile << "blockConfig" << paramSeparator << "dynamicSharedMemorySize >>> ";
		
		// scalar parameters and metadata arguments 
		programFile << paramIndent << indentStr.str() << "(";
		programFile << "arrayMetadata" << paramSeparator;
		programFile << "partition" << paramSeparator << paramIndent << indentStr.str();
		programFile << "taskGlobalsGpu" << paramSeparator << "threadLocalsGpu";
		programFile << paramSeparator << "stageExecutionTrackerGpu";
		programFile << paramSeparator << paramIndent << indentStr.str();
		programFile << "launchMetadata" << paramSeparator;
		programFile << "hostLpuConfigs" << paramSeparator << "maxPartSizes";

		// buffer referrence arguments for array variables
		for (int j = 0; j < accessedArrays->NumElements(); j++) {
			programFile << paramSeparator << paramIndent << indentStr.str();
			programFile << "*" << accessedArrays->Nth(j) << "Buffers";
		}
		programFile << ")" << stmtSeparator << std::endl;
	}
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

//-------------------------------------------------------- Variable Locality Spec ----------------------------------------------------------/

GpuVarLocalitySpec::GpuVarLocalitySpec(const char *vN, Space *aLps, bool sCS, bool rPWI) {
	varName = vN;
	allocatingLps = aLps;
	smLocalCopySupported = sCS;
	reqPerWarpInstances = rPWI;
}

void GpuVarLocalitySpec::describe(int indentLevel) {
	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << '\t';
	std::cout << indentStr.str() << " Variable: " << varName << "\n";
	indentStr << '\t';
	std::cout << indentStr.str() << "Needed in Space: " << allocatingLps->getName() << '\n';
	std::cout << indentStr.str();
	if (smLocalCopySupported) std::cout << "Can be stored in SM\n";
	else std::cout << "Cannot be stored in SM\n";
	std::cout << indentStr.str();
	if (reqPerWarpInstances) std::cout << "Each warp will have personal copy\n";
	else std::cout << "All warps should share a single copy\n";
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
	this->varAllocInstrList = NULL;
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

List<GpuVarLocalitySpec*> *GpuExecutionContext::filterVarAllocInstrsForLps(Space *lps) {
	
	List<GpuVarLocalitySpec*> *filteredInstrs = new List<GpuVarLocalitySpec*>;
	for (int i = 0; i < varAllocInstrList->NumElements(); i++) {
		GpuVarLocalitySpec *instr = varAllocInstrList->Nth(i);
		if (instr->getAllocatingLps() == lps) {
			filteredInstrs->Append(instr);
		}
	}
	return filteredInstrs;
}

List<GpuVarLocalitySpec*> *GpuExecutionContext::filterModifiedVarAllocInstrsForLps(Space *lps) {
	
	List<GpuVarLocalitySpec*> *allocFilter = filterVarAllocInstrsForLps(lps);
	List<GpuVarLocalitySpec*> *modificationFilter = new List<GpuVarLocalitySpec*>;
	List<const char*> *modifiedVars = getModifiedVariableList();	

	for (int i = 0; i < allocFilter->NumElements(); i++) {
		
		GpuVarLocalitySpec *instr = allocFilter->Nth(i);

		// if the data update has been done at the GPU card memory directly then there is no need for
		// a separate synchronization of local SM update to global memory
		if (!instr->isSmLocalCopySupported()) continue;

		if (string_utils::contains(modifiedVars, instr->getVarName())) {
			modificationFilter->Append(instr);
		}
	}

	return modificationFilter;
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
			} else {
				currentSubflow->Append(stage);
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

void GpuExecutionContext::analyzeVarAllocReqs(PartitionHierarchy *lpsHierarchy) {

	varAllocInstrList = new List<GpuVarLocalitySpec*>;
	List<const char*> *accessedVariables = getVariableAccessList();
	List<ExecutionStage*> *execStageList = getComputeStagesOfFlowContext();
	int gpuPpsId = lpsHierarchy->getPCubeSModel()->getGpuTransitionSpaceId();
	int smPpsId = gpuPpsId - 1;
	Space *rootLps = lpsHierarchy->getRootSpace();

	for (int i = 0; i < accessedVariables->NumElements(); i++) {

		// Scalar variables are kept on the GPU card memory; so allocation analysis is done for arrays only
		const char *varName = accessedVariables->Nth(i);
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array == NULL) continue;

		Space *varNeedingLps = getEarliestLpsNeedingVar(varName, execStageList, lpsHierarchy);
		int ppsId = varNeedingLps->getPpsId();
		if (ppsId > smPpsId) {
			GpuVarLocalitySpec *varLocSpec = new GpuVarLocalitySpec(varName, 
					varNeedingLps, false, false);
			varAllocInstrList->Append(varLocSpec);
		} else {
			Space *allocLps = getInnermostSMLpsForVarCopy(varName, smPpsId, varNeedingLps);
			
			// This is an safety check that ensures that once it is known that the array data parts can 
			// be stored inside GPU SMs, the allocator LPS is not mapped above it. Although this is not
			// supposed to be allowed by the getInnermostSMLpsForVarCopy function, it can happen if the
			// context LPS for the GPU sub-flow is a sub-partition LPS and the array is partitioned only
			// in the parent LPS not in the context LPS.
			if (contextLps->isParentSpace(allocLps)) allocLps = contextLps;

			ppsId = allocLps->getPpsId();
			GpuVarLocalitySpec *varLocSpec = NULL;
			if (ppsId == smPpsId) {
				varLocSpec = new GpuVarLocalitySpec(varName, allocLps, true, false);
			} else {
				varLocSpec = new GpuVarLocalitySpec(varName, allocLps, true, true);
			}
			varAllocInstrList->Append(varLocSpec);
		}
	}

	// We haven't figured out the logic for copying data from GPU card memory data part to SM memory smaller data
	// part when there was some index reordering in-between the 2 LPSes holding those two parts. So for now we 
	// exit with an error if such reordering is detected.
	for (int i = 0; i < varAllocInstrList->NumElements(); i++) {
		GpuVarLocalitySpec *instr = varAllocInstrList->Nth(i);
		Space *allocLps = instr->getAllocatingLps();
		const char *varName = instr->getVarName();	
		ArrayDataStructure *smStruct = (ArrayDataStructure*) allocLps->getStructure(varName);
		if (smStruct->isReordered(contextLps)) {
			std::cout << "At this moment the compiler cannot handle the case where the array indices\n"; 
			std::cout << "are reordered between the topmost LPS mapped to the GPU and the LPS mapped\n";
			std::cout << "inside SMs. Array '" << varName << "' has its index reordered in-between\n";
			std::cout << "Space " << contextLps->getName() << " and Space " << allocLps->getName() << "\n";
			std::exit(EXIT_FAILURE);
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

	// if applicable, update stage execution counters related to the GPU sub-flow context that will be needed 
	// to activate data synchronizers and communicators
	List<ExecutionStage*> *execStageList = getComputeStagesOfFlowContext();
	for (int i = 0; i < execStageList->NumElements(); i++) {
		ExecutionStage *execStage =execStageList->Nth(i);
		StageSyncReqs *syncReqs = execStage->getAllSyncRequirements();
		List<SyncRequirement*> *syncList = syncReqs->getAllSyncRequirements();
		for (int i = 0; i < syncList->NumElements(); i++) {
			DependencyArc *arc = syncList->Nth(i)->getDependencyArc();
			if (arc->doesRequireSignal()) {
				stream << indent.str() << arc->getArcName();
				stream << " += gpuCodeExecutor->getExecutionCount(\"";
				stream << execStage->getName() << "ExecutionCounter";
				stream << "\")" << stmtSeparator;
			}
		}
	}

	stream << indent.str() << "} // GPU LPU offloading context ends\n";
}

List<ExecutionStage*> *GpuExecutionContext ::getComputeStagesOfFlowContext() {
	
	List<ExecutionStage*> *executeStageList = new List<ExecutionStage*>;
	std::deque<FlowStage*> stageQueue;
	for (int i = 0; i < contextFlow->NumElements(); i++) {
		stageQueue.push_back(contextFlow->Nth(i));
	}

	while (!stageQueue.empty()) {
		FlowStage *currentStage = stageQueue.front();
		stageQueue.pop_front();
		if (dynamic_cast<ExecutionStage*>(currentStage) != NULL) {
			executeStageList->Append((ExecutionStage*) currentStage);
			continue;
		}
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(currentStage);
		if (compositeStage == NULL) continue;
		List<FlowStage*> *nestedStages = compositeStage->getStageList();
		for (int i = 0; i < nestedStages->NumElements(); i++) {
			stageQueue.push_back(nestedStages->Nth(i));
		} 
	}

	return executeStageList;
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

void GpuExecutionContext::generateGpuKernel(CompositeStage *kernelDef, 
                std::ofstream &programFile, PCubeSModel *pcubesModel) {
	
	int gpuPpsLevel = pcubesModel->getGpuTransitionSpaceId();
	int contextPps = contextLps->getPpsId();
	const char *contextLpsName = contextLps->getName();
	List<const char*> *arrayNames = contextLps->getLocallyUsedArrayNames();
        List<const char*> *accessedArrays 
			= string_utils::intersectLists(getVariableAccessList(), arrayNames);
	bool gpuLevel = (contextPps == gpuPpsLevel);
	bool smLevel = (contextPps == gpuPpsLevel - 1);
	bool warpLevel = (contextPps == gpuPpsLevel - 2);
	GpuCodeConstants *gpuCons = NULL;
	if (gpuLevel) gpuCons = GpuCodeConstants::getConstantsForGpuLevel();
	else if (smLevel) gpuCons = GpuCodeConstants::getConstantsForSmLevel();
	else gpuCons = GpuCodeConstants::getConstantsForWarpLevel();

	/**************************************************************************************************************
		     Assign shared memory data part referrences for variables from the memory panel.
	***************************************************************************************************************/

        // first do the declarations
        programFile << indent << "// global and shared memory data parts holders' declarations\n";
	for (int i = 0; i < varAllocInstrList->NumElements(); i++) {
		GpuVarLocalitySpec *varSpec = varAllocInstrList->Nth(i);
		const char *arrayName = varSpec->getVarName();
                ArrayDataStructure *array = (ArrayDataStructure*) contextLps->getLocalStructure(arrayName);
                ArrayType *arrayType = (ArrayType*) array->getType();
                Type *elemType = arrayType->getTerminalElementType();
                programFile << indent << "__shared__ " << elemType->getCType();
		if (varSpec->doesReqPerWarpInstances()) {
			programFile << " *" << arrayName << "[WARP_COUNT]" << paramSeparator;
			programFile << " *" << arrayName << "_global[WARP_COUNT]" << stmtSeparator;
		} else {
			programFile << " *" << arrayName << paramSeparator;
			programFile << " *" << arrayName << "_global" << stmtSeparator;
		}
	}
        programFile << std::endl;

	// then do the assignment of indices from the shared memory panel to the shared data part holder refereces; 
	// a single thread does this even if warp mapping is used to ensure that the memory panel index has been 
	// advanced properly 
        programFile << indent << "// shared memory data parts holders allocation\n";
        programFile << indent << "if (warpId == 0 && threadId == 0) {\n";
        programFile << doubleIndent << "panelIndex = 0" << stmtSeparator;
	for (int i = 0; i < varAllocInstrList->NumElements(); i++) {
		GpuVarLocalitySpec *varSpec = varAllocInstrList->Nth(i);
		const char *arrayName = varSpec->getVarName();
                ArrayDataStructure *array = (ArrayDataStructure*) contextLps->getLocalStructure(arrayName);
                ArrayType *arrayType = (ArrayType*) array->getType();
                Type *elemType = arrayType->getTerminalElementType();
		std::string indentLevel = std::string(doubleIndent);
                if (varSpec->doesReqPerWarpInstances()) {
			programFile << doubleIndent << "for (int i = 0; i < WARP_COUNT; i++) {\n";
			indentLevel = std::string(tripleIndent);
		}		
		programFile << indentLevel << arrayName;
		if (varSpec->doesReqPerWarpInstances())  programFile << "[i]"; 
		programFile<< " = (" << elemType->getCType() << "*) ";
		programFile << "(memoryPanel + panelIndex)" << stmtSeparator;
		programFile << indentLevel << "panelIndex += " << "maxPartSizes." << arrayName;
		programFile << "MaxPartSize" << stmtSeparator;		
		if (varSpec->doesReqPerWarpInstances()) programFile << doubleIndent << "}\n";
	}

        programFile << indent << "}\n";
        programFile << indent << "__syncthreads()" << stmtSeparator;
        programFile << std::endl;

	/**************************************************************************************************************
			declare local and shared metadata structures for LPU part dimensions tracking
	***************************************************************************************************************/
	
	// if the GPU context LPS is above the warp level then there will be one structure per variable, otherwise
	// there will be one per warp
	programFile << indent << "// metadata declaration for holding Topmost LPU storage dimensions\n";
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		const char *arrayName = accessedArrays->Nth(i);
                ArrayDataStructure *array = (ArrayDataStructure*) contextLps->getLocalStructure(arrayName);
		int dimensions = array->getDimensionality();
		programFile << indent << "__shared__ GpuDimension " << arrayName << "SGRanges";
		programFile << gpuCons->storageSuffix << "[" << dimensions << "]" << paramSeparator;
		programFile << arrayName << "Space" << contextLpsName << "PRanges";
		programFile << gpuCons->storageSuffix << "[" << dimensions << "]" << stmtSeparator;
	}
	programFile << std::endl;

	// declare a tracker object for storing hierarchical array dimension partition information during index
	// transformation of arrays using index reordering partition function 
	programFile << indent << "// index transformation helper variable\n";
	programFile << indent << "GpuDimPartConfig partConfig" << stmtSeparator;
	programFile << std::endl;
	
	// declare an integer to hold intermediate values of a reordered index during the transformation process 
	programFile << indent << "// create a local transformed index variable for later use\n";
        programFile << indent << "int xformIndex" << stmtSeparator;
	programFile << std::endl;

	/**************************************************************************************************************
				distribute the top level LPUs staged-in from the host
	***************************************************************************************************************/
	
	programFile << indent << "{ // scope starts for distribution of staged-in LPUs\n\n";
	generateStagedInLpuDistributionLoop(programFile, pcubesModel);

	// GPU or SM level LPU iterations will always be preceeded by a syncthreads to ensure that all warps have
	// completed previous LPU processing
	if (gpuLevel || smLevel) {
		programFile << std::endl << doubleIndent;
		programFile << "// ensure all warps finished processing the last LPU\n";
		programFile << doubleIndent << "__syncthreads()" << stmtSeparator;
		programFile << std::endl;
	}

	/**************************************************************************************************************
			  data part reference and metadata initialization from GPU card memory
	***************************************************************************************************************/
	
	decorator::writeCommentHeader(2, &programFile, "kernel argument buffers processing start");
	
	// before processing can start on any LPU, we need to read dimension information of all component data 
	// parts and index into the appropriate positions of the card memory LPU data buffer so that data copying
	// can be done from the global card memory to the SM shared memory
	if (gpuLevel || smLevel) {
		programFile << doubleIndent << "if (warpId == 0 && threadId == 0) {\n";	
	} else {
		programFile << doubleIndent << "if (threadId == 0) {\n";
	} 
	programFile << std::endl;

	// declare some auxiliary metadata to be used during interpreting GPU card memory buffers
	programFile << tripleIndent << "// auxiliary variables for reading data buffers\n";
	programFile << tripleIndent << "__shared__ int lpuIndex";
	programFile << gpuCons->storageSuffix << stmtSeparator;
	programFile << tripleIndent << "__shared__ int partIndex";
	programFile << gpuCons->storageSuffix << stmtSeparator;
	programFile << tripleIndent << "__shared__ int partStartPos";
	programFile << gpuCons->storageSuffix << stmtSeparator;
	programFile << tripleIndent << "__shared__ int partDimRangeStart";
	programFile << gpuCons->storageSuffix << stmtSeparator;
	programFile << tripleIndent << "__shared__ int partRangeDepth";
	programFile << gpuCons->storageSuffix << stmtSeparator;
	programFile << std::endl;

	// determine the current LPU's index for accessing data buffers
	programFile << tripleIndent << "lpuIndex " << gpuCons->storageIndex;
	programFile << " = launchMetadata.entries[";
	if (contextType == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {
		programFile << gpuCons->distrIndex << "].batchStartIndex " << paramIndent;
		programFile << tripleIndent << " + space" << contextLpsName << "LinearId "; 
		programFile << "- launchMetadata.entries[" << gpuCons->distrIndex;
		programFile << "].batchRangeMin" << stmtSeparator;
	} else {
		programFile << "0].batchStartIndex" << paramIndent;
		programFile << tripleIndent << " + space" << contextLpsName << "LinearId "; 
		programFile << "- launchMetadata.entries[0].batchRangeMin" << stmtSeparator;
	}

	// iterate over the accessed arrays one by one to initialize their data pointers and metadata
	for (int i = 0; i < accessedArrays->NumElements(); i++) {
		
		const char *varName = accessedArrays->Nth(i);
                ArrayDataStructure *array = (ArrayDataStructure*) contextLps->getLocalStructure(varName);
		int dimensions = array->getDimensionality();
                ArrayType *arrayType = (ArrayType*) array->getType();
                Type *elemType = arrayType->getTerminalElementType();

		programFile << std::endl << tripleIndent;
		programFile << "// retrieving variable '" << varName << "' information\n";
		programFile << tripleIndent << "partIndex" << gpuCons->storageIndex << " = " << varName;
		programFile << "Buffers.partIndexBuffer[lpuIndex";
		programFile << gpuCons->storageIndex << "]" << stmtSeparator;
		programFile << tripleIndent << "partStartPos";
		programFile << gpuCons->storageIndex << " = " << varName;
		programFile << "Buffers.partBeginningBuffer[partIndex" << gpuCons->storageIndex << "]";
		programFile << stmtSeparator;
		programFile << tripleIndent << varName << "_global" << gpuCons->storageIndex << " = ";
		programFile << "(" << elemType->getCType() << "*) ";
		programFile << "(" << varName << "Buffers.dataBuffer + partStartPos";
		programFile << gpuCons->storageIndex << ")" << stmtSeparator;
		programFile << tripleIndent << "partRangeDepth" << gpuCons->storageIndex << " = (" << varName;
		programFile << "Buffers.partRangeDepth + 1) * " << dimensions << " * 2" << stmtSeparator;
		programFile << tripleIndent << "partDimRangeStart" << gpuCons->storageIndex << " = ";
		programFile << "partRangeDepth" << gpuCons->storageIndex;
		programFile << " * partIndex" << gpuCons->storageIndex << stmtSeparator;

		programFile << std::endl;
		for (int j = 0; j < dimensions; j++) {
			programFile << tripleIndent << varName << "SGRanges" << gpuCons->storageIndex;
			programFile << "[" << j << "].range.min = ";
			programFile << varName << "Space" << contextLpsName;
			programFile << "PRanges" << gpuCons->storageIndex << "[" << j << "].range.min";
			programFile << "\n" << tripleIndent << doubleIndent;
			programFile << " = " << varName << "Buffers.partRangeBuffer[";
			programFile << "partDimRangeStart" << gpuCons->storageIndex << " + ";
			programFile << 2 * j << "]" << gpuCons->storageIndex << stmtSeparator;
			programFile << tripleIndent << varName << "SGRanges" << gpuCons->storageIndex;
			programFile << "[" << j << "].range.max = ";
			programFile << varName << "Space" << contextLpsName;
			programFile << "PRanges" << gpuCons->storageIndex << "[" << j << "].range.max";
			programFile << "\n" << tripleIndent << doubleIndent;
			programFile << " = " << varName << "Buffers.partRangeBuffer[";
			programFile << "partDimRangeStart" << gpuCons->storageIndex << " + ";
			programFile << 2 * j + 1 << "]" << gpuCons->storageIndex << stmtSeparator;
		}
	}	

	// end of global memory buffer processing loop
	programFile << doubleIndent << "}\n";
	if (gpuLevel || smLevel) {
		programFile << doubleIndent << "__syncthreads()" << stmtSeparator;
	} 
	
	decorator::writeCommentHeader(2, &programFile, "kernel argument buffers processing end");
	
	/**************************************************************************************************************
					      Card to SM Memory Data Stage In
	***************************************************************************************************************/
	
	List<GpuVarLocalitySpec*> *allocInstrList = filterVarAllocInstrsForLps(contextLps);
        if (allocInstrList->NumElements() > 0) {
		const char *stageInIndent = "\t\t";
		CompositeStage *dummyStage = new CompositeStage(0, contextLps, NULL);
                dummyStage->generateCardToSmDataStageIns(programFile, 
				stageInIndent, this, 
				gpuPpsLevel, allocInstrList);
        }

	/**************************************************************************************************************
					    Generate CUDA Code for the Sub-Flow
	***************************************************************************************************************/

	List<FlowStage*> *kernelStages = kernelDef->getStageList();
	for (int i = 0; i < kernelStages->NumElements(); i++) {
		FlowStage *stage = kernelStages->Nth(i);
		stage->generateGpuKernelCode(programFile, 2, this, contextLps, gpuPpsLevel);
	}

	/**************************************************************************************************************
					      SM Memory to Card Data Stage Out
	***************************************************************************************************************/
	
	List<GpuVarLocalitySpec*> *updateInstrList = filterModifiedVarAllocInstrsForLps(contextLps);
        if (updateInstrList->NumElements() > 0) {
		const char *stageOutIndent = "\t\t";
		CompositeStage *dummyStage = new CompositeStage(0, contextLps, NULL);
                dummyStage->generateSmToCardDataStageOuts(programFile,
                                stageOutIndent, this,
                                gpuPpsLevel, updateInstrList);
        }

	// end of outer most LPU iteration loop	
	programFile << indent << "}\n\n";
	programFile << indent << "} // scope ends for distribution of staged-in LPUs\n";
}

void GpuExecutionContext::generateContextFlowImplementerCode(std::ofstream &programFile, int indentLevel) {
	
	List<const char*> *arrayNames = contextLps->getLocallyUsedArrayNames();
        List<const char*> *accessedArrays 
			= string_utils::intersectLists(getVariableAccessList(), arrayNames);

	for (int i = 0; i < kernelConfigList->NumElements(); i++) {
		KernelGroupConfig *kernelGroup = kernelConfigList->Nth(i);
		kernelGroup->generateKernelGroupExecutionCode(programFile, accessedArrays, indentLevel);
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
	wrapOffloadingCodeInLargerContext(stream, 
			indentation + 1, transitLpsList, index + 1, offloadingCode);

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

const char *GpuExecutionContext::generateDataCopyingLoopHeaders(std::ofstream &stream,
		ArrayDataStructure *array, 
		int indentLevel, bool warpLevel) {
		
	const char *varName = array->getName();
	int dimensions = array->getDimensionality();
	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel; i++) indentStr << indent;	
	
	if (warpLevel) {
		
		// all threads iterate over the points in the index ranges of the higher dimenions
		int i = 0; 
		for (; i < dimensions - 1; i++) {
			stream << indentStr.str() << "for (int d" << "_" << i << " = ";
			stream << varName << "SRanges[warpId][" << i << "].range.min; ";
			stream << paramIndent << indentStr.str();
			stream << "d_" << i << " <= " << varName;
			stream << "SRanges[warpId][" << i << "].range.max;";
			stream << " d_" << i << "++) {\n";
			indentStr << indent;	
		}
		// the lowermost dimension's index range is distributed among the threads
		stream << indentStr.str() << "for (int d" << "_" << i << " = ";
		stream << varName << "SRanges[warpId][" << i << "].range.min + threadId; ";
		stream << paramIndent << indentStr.str();
		stream << "d_" << i << " <= SRanges[warpId][" << i << "].range.max; d_";
		stream << i << " += WAPR_SIZE) {\n";
		indentStr << indent;	

		return strdup(indentStr.str().c_str());
	}

	// if the copying is being done for the entire SM then a distinction has been made between single 
	// dimensional and multidimensional arrays

	// for single dimensional arrays, the single dimension gets distributed to all threads of the SM
	if (dimensions == 1) {
		stream << indentStr.str() << "for (int d_0 = ";
		stream << varName << "SRanges[0].range.min + threadIdx.x; ";
		stream << paramIndent << indentStr.str();
		stream << "d_0 <= " << varName << "SRanges[0].range.max;";
		stream << " d_0 += WARP_SIZE * WARP_COUNT) {\n";
		indentStr << indent;	
		return strdup(indentStr.str().c_str());
	}

	// for multidimensional arrays, first let all threads iterate over the all but last two dimensions
	// of the array
	int i = 0; 
	for (; i < dimensions - 2; i++) {
		stream << indentStr.str() << "for (int d" << "_" << i << " = ";
		stream << varName << "SRanges[" << i << "].range.min; ";
		stream << paramIndent << indentStr.str();
		stream << "d_" << i << " <= " << varName << "SRanges[" << i << "].range.max;";
		stream << " d_" << i << "++) {\n";
		indentStr << indent;	
	}

	// distribute indexes in the next to last dimension to the warps and indexes in the last dimension
	// to the threads of a warp
	stream << indentStr.str() << "for (int d" << "_" << i << " = ";
	stream << varName << "SRanges[" << i << "].range.min + warpId; ";
	stream << paramIndent << indentStr.str();
	stream << "d_" << i << " <= " << varName << "SRanges[" << i << "].range.max;";
	stream << " d_" << i << " += WARP_COUNT) {\n";
	indentStr << indent;
	i++;	
	stream << indentStr.str() << "for (int d" << "_" << i << " = ";
	stream << varName << "SRanges[" << i << "].range.min + threadId; ";
	stream << paramIndent << indentStr.str();
	stream << "d_" << i << " <= " << varName << "SRanges[" << i << "].range.max;";
	stream << " d_" << i << " += WARP_SIZE) {\n";
	indentStr << indent;
	return strdup(indentStr.str().c_str());
}

void GpuExecutionContext::generateElementTransferStmt(std::ofstream &stream,           
		ArrayDataStructure *array,
		const char *indentPrefix, 
		bool warpLevel, int transferDirection) {
	
	const char *varName = array->getName();
	int dimensions = array->getDimensionality();
	
	std::stringstream sender, receiver;
	std::stringstream sendRange, recvRange;
	if (transferDirection == 1) {
		sender << varName << "_global";
		sendRange << varName << "SGRanges";	
		receiver << varName;
		recvRange << varName << "SRanges";
	} else {
		sender << varName;
		sendRange << varName << "SRanges";
		receiver << varName << "_global";
		recvRange << varName << "SGRanges";
	}
	if (warpLevel) {
		sender << "[warpId]";
		sendRange << "[warpId]";
		receiver << "[warpId]";
		recvRange << "[warpId]";
	}
	
	stream << indentPrefix << receiver.str() << "[";
	for (int i = 0; i < dimensions; i++) {
		if (i > 0) {
			stream << paramIndent << indentPrefix << " + ";
		}
		stream << "(d_" << i << " - " << recvRange.str() << "[" << i << "].range.min)";
		if (i < dimensions - 1) {
			for (int j = i + 1; j < dimensions; j++) {
				stream << paramIndent << indentPrefix << " * ";
				stream << "(" << recvRange.str() << "[" << j << "].range.max - ";
				stream << recvRange.str() << "[" << j << "].range.min + 1)";
			}
		}	
	}
	stream << "]\n" << indentPrefix << indent;
	stream << "= " << sender.str() << "[";
	for (int i = 0; i < dimensions; i++) {
		if (i > 0) {
			stream << paramIndent << indentPrefix << " + ";	
		}
		stream << "(d_" << i << " - " << sendRange.str() << "[" << i << "].range.min)";
		if (i < dimensions - 1) {
			for (int j = i + 1; j < dimensions; j++) {
				stream << paramIndent << indentPrefix << " * ";
				stream << "(" << sendRange.str() << "[" << j << "].range.max - ";
				stream << sendRange.str() << "[" << j << "].range.min + 1)";
			}
		}	
	}
	stream << "]" << stmtSeparator;	
}

Space *GpuExecutionContext::getEarliestLpsNeedingVar(const char *varName, 
		List<ExecutionStage*> *execStageList, PartitionHierarchy *lpsHierarchy) {

	Space *currentLps = NULL;
	for (int i = 0; i < execStageList->NumElements(); i++) {
		ExecutionStage *stage = execStageList->Nth(i);
		VariableAccess *varAccess =  stage->getAccessMap()->Lookup(varName);
		if (varAccess == NULL) continue;
		if (!varAccess->isContentAccessed()) continue;
		Space *stageLps = stage->getSpace();
		if (currentLps == NULL) {
			currentLps = stageLps;
		} else {
			currentLps = lpsHierarchy->getCommonAncestor(currentLps, stageLps);
		}
	}

	// The selected LPS must have the variable being partitioned in it. otherwise, return the closest ancestor
	// LPS that has the variable
	DataStructure *variable = NULL;
	while ((variable = currentLps->getLocalStructure(varName)) == NULL) {
		currentLps = currentLps->getParent();
	}

	// The final twist here is for subpartitioned LPSes. If the variable is subpartitioned then we need to 
	// return the subpartition LPS; otherwise, the parent LPS of the subpartition should be returned. To have
	// this done appropriately we just get and return the LPS from the variable.
	if (currentLps == contextLps) return currentLps;
	else return variable->getSpace();
}

Space *GpuExecutionContext::getInnermostSMLpsForVarCopy(const char *varName, 
                        int smPpsId, Space *earliestLpsNeedingVar) {

	// currently this analysis is done for arrays only; so we can safely type-cast the variable to an array
	ArrayDataStructure *array = (ArrayDataStructure *) earliestLpsNeedingVar->getLocalStructure(varName);
	
	Space *selectedLps = earliestLpsNeedingVar;
	bool warpLevel = (smPpsId - earliestLpsNeedingVar->getPpsId()) == 1;
	if (warpLevel) {
		while (selectedLps != contextLps) {
			ArrayDataStructure *source = (ArrayDataStructure*) array->getSource();
			Space *parentLps = source->getSpace();
			int parentPpsId = parentLps->getPpsId();
			if (parentPpsId > smPpsId) {
				// if we are crossing the SM memory boundary when trying to uplift variable copy 
				// then there is no hope
				return selectedLps;
			} else if (parentPpsId == smPpsId) {
				// if the parent is at the SM level we have found our LPS to avoid separate warp
				// level data part instances
				selectedLps = parentLps;
				break;
			} else {
				// if the parent of the current warp level LPS is also mapped to the warp level 
				// then we need to check if uplifting the copy operation to the parent LPS 
				// increases the memory requirement
				if (source->isPartitioned()) return selectedLps;
				selectedLps = parentLps;
				array = source;
			}
		}
	}

	// attempt data copy uplifting at the SM level
	array = (ArrayDataStructure *) selectedLps->getLocalStructure(varName);
	while (selectedLps != contextLps) {
		ArrayDataStructure *source = (ArrayDataStructure *) array->getSource();
		Space *parentLps = source->getSpace();
		int parentPpsId = parentLps->getPpsId();
		if (parentPpsId > smPpsId || source->isPartitioned()) return selectedLps;
		array = source;
		selectedLps = parentLps;
	}
	return selectedLps;
}

void GpuExecutionContext::generateStagedInLpuDistributionLoop(std::ofstream &programFile, PCubeSModel *pcubesModel) {
	
	int gpuPpsLevel = pcubesModel->getGpuTransitionSpaceId();
        int contextPps = contextLps->getPpsId();
        const char *contextLpsName = contextLps->getName();
        bool gpuLevel = (contextPps == gpuPpsLevel);
        bool smLevel = (contextPps == gpuPpsLevel - 1);
        bool warpLevel = (contextPps == gpuPpsLevel - 2);
        GpuCodeConstants *gpuCons = NULL;
        if (gpuLevel) gpuCons = GpuCodeConstants::getConstantsForGpuLevel();
        else if (smLevel) gpuCons = GpuCodeConstants::getConstantsForSmLevel();
        else gpuCons = GpuCodeConstants::getConstantsForWarpLevel();
        int lpsDimension = contextLps->getDimensionCount();


	/**************************************************************************************************************
					Distribution of LPUs of an un-partitioned LPS
	***************************************************************************************************************/
	
	if (lpsDimension == 0) {
		programFile << indent << "for (int space" << contextLpsName; 
		programFile << "LinearId = launchMetadata.entries[0]";
		programFile << ".batchRangeMin + " << gpuCons->distrIndex << "; ";
		programFile << paramIndent << indent << "space" << contextLpsName;
		programFile << "LinearId <= launchMetadata.entries[0].batchRangeMax; ";
		programFile << "space" << contextLpsName << "LinearId += " << gpuCons->jumpExpr << ") {\n";
		return;
	}	
	
	/**************************************************************************************************************
				      LPU Counter Initialization for an Partitioned LPS
	***************************************************************************************************************/

	// first determine the LPU counts along different dimensions
	if (contextType == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {
		programFile << indent << "__shared__ int space" << contextLpsName << "LpuCount";
		programFile << gpuCons->storageSuffix << "[" << lpsDimension << "]" << stmtSeparator;
		if (gpuLevel || smLevel) {
			programFile << indent << "if (warpId == 0 && threadId == 0) {\n";
		} else {
			programFile << indent << "if (threadId == 0) {\n";
		}
		for (int i = 0; i < lpsDimension; i++) {
			programFile << doubleIndent << "space" << contextLpsName << "LpuCount";
			programFile << gpuCons->storageIndex << "[";
			programFile << i << "] = launchMetadata.entries[" << gpuCons->distrIndex << "].";
			programFile << "lpuCount" << i + 1 << stmtSeparator;
		}
		programFile << indent << "}\n";
		programFile << indent << "__syncthreads()" << stmtSeparator << std::endl;
	} else {
		programFile << indent << "__shared__ int space" << contextLpsName << "LpuCount";
		programFile << "[" << lpsDimension << "]" << stmtSeparator;
		programFile << indent << "if (warpId == 0 && threadId == 0) {\n";
		for (int i = 0; i < lpsDimension; i++) {
			programFile << doubleIndent << "space" << contextLpsName << "LpuCount";
			programFile << "[" << i << "] = launchMetadata.entries[0].";
			programFile << "lpuCount" << i + 1 << stmtSeparator;
		}
		programFile << indent << "}\n";
		programFile << indent << "__syncthreads()" << stmtSeparator << std::endl;
	}

	/**************************************************************************************************************
					 Distribution of LPUs of a partitioned LPS
	***************************************************************************************************************/

	// when the context is location sensitive make the PPU iterate over their respective batch ranges
	if (contextType == LOCATION_SENSITIVE_LPU_DISTR_CONTEXT) {

		programFile << indent << "for (int space" << contextLpsName;
		programFile << "LinearId = launchMetadata.entries[" << gpuCons->distrIndex << "]";
		programFile << ".batchRangeMin; " << paramIndent << indent;
		programFile << "space" << contextLpsName;
		programFile << "LinearId <= launchMetadata.entries[" << gpuCons->distrIndex << "].batchRangeMax; ";
		programFile << "space" << contextLpsName;
		programFile << "LinearId++) {\n\n";
		
		// if the batch range for the current PPU starts with an invalid identifier then the PPU does not
		// participate in any computation
		programFile << doubleIndent  << "// exiting if the batch range is invalid\n";
		programFile << doubleIndent << "if (space";
		programFile << contextLpsName << "LinearId == INVALID_ID) break" << stmtSeparator;
	
	// otherwise let the PPUs stride over the single range of LPUs
	} else {
		programFile << indent << "for (int space" << contextLpsName; 
                programFile << "LinearId = launchMetadata.entries[0]";
                programFile << ".batchRangeMin + " << gpuCons->distrIndex << "; ";
                programFile << paramIndent << indent << "space" << contextLpsName;
                programFile << "LinearId <= launchMetadata.entries[0].batchRangeMax; ";
                programFile << "space" << contextLpsName << "LinearId += " << gpuCons->jumpExpr << ") {\n";
	}	
}
