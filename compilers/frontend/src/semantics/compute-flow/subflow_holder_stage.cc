#include "../computation_flow.h"
#include "../scope.h"
#include "../symbol.h"
#include "../task_space.h"
#include "../data_access.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_expr.h"
#include "../../syntax/ast_stmt.h"
#include "../../syntax/ast_task.h"
#include "../../static-analysis/reduction_info.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <iostream>
#include <fstream>

//------------------------------------------------------- Repeat Control Block --------------------------------------------------/

RepeatControlBlock::RepeatControlBlock(Space *space, 
		RepeatCycleType type, Expr *executeCond) : CompositeStage(space) {
	this->type = type;
	this->condition = executeCond;
}

void RepeatControlBlock::performDataAccessChecking(Scope *taskScope) {
	accessMap = validateDataAccess(taskScope, condition, NULL);
	CompositeStage::performDataAccessChecking(taskScope);
}

void RepeatControlBlock::print(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << "Repition: ";
	std::cout << "(Space " << space->getName() << ") ";
	std::cout << "[" << index << "," << groupNo << "," << repeatIndex << "]\n";
	CompositeStage::print(indentLevel);
}

int RepeatControlBlock::assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle) {

        this->index = currentIndex;
        this->groupNo = currentGroupNo;
        this->repeatIndex = currentRepeatCycle;

        int nextIndex = currentIndex + 1;
        int nextRepeatIndex = currentRepeatCycle + 1;

        for (int i = 0; i < stageList->NumElements(); i++) {
                FlowStage *stage = stageList->Nth(i);
                nextIndex = stage->assignIndexAndGroupNo(nextIndex, this->index, nextRepeatIndex);
        }
        return nextIndex;
}

void RepeatControlBlock::calculateLPSUsageStatistics() {
	
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        
	while ((accessLog = iterator.GetNextValue()) != NULL) {
                if (!accessLog->isContentAccessed()) continue;
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getLocalStructure(varName);
                AccessFlags *accessFlags = accessLog->getContentAccessFlags();
                LPSVarUsageStat *usageStat = structure->getUsageStat();

                // As repeat condition is supposed to be evaluated multiple times, any data structure been used
                // here should be marked as multiple access. Thus we add access information twice.
                if (accessFlags->isRead() || accessFlags->isWritten()) {
                        usageStat->addAccess();
                        usageStat->addAccess();
                }
        }

        // Two calls have been made for evaluating the usage statistics in nested computation stages assuming 
	// that a repeat cycle will executes at least twice, and two calls are sufficient to flag multiple 
	// accesses to data structures nested within the repeat block.
        CompositeStage::calculateLPSUsageStatistics();
        CompositeStage::calculateLPSUsageStatistics();
}

void RepeatControlBlock::performEpochUsageAnalysis() {
        CompositeStage::performEpochUsageAnalysis();
        FlowStage::CurrentFlowStage = this;
        condition->setEpochVersions(space, 0);
}

void RepeatControlBlock::setLpsExecutionFlags() {
        CompositeStage::setLpsExecutionFlags();
        if (this->isLpsDependent()) {
                space->flagToExecuteCode();
        }
}

void RepeatControlBlock::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
        FlowStage::fillInTaskEnvAccessList(envAccessList);
        CompositeStage::fillInTaskEnvAccessList(envAccessList);
}

void RepeatControlBlock::prepareTaskEnvStat(TaskEnvStat *taskStat) {
        FlowStage::prepareTaskEnvStat(taskStat);
        CompositeStage::prepareTaskEnvStat(taskStat);
}

List<ReductionMetadata*> *RepeatControlBlock::upliftReductionInstrs() {
	List<ReductionMetadata*> *upliftedReductions = CompositeStage::upliftReductionInstrs();
	for (int i = 0; i < upliftedReductions->NumElements(); i++) {
		ReductionMetadata *metadata = upliftedReductions->Nth(i);
		const char *variable = metadata->getResultVar();
		const char *reductionRootLps = metadata->getReductionRootLps()->getName();
		ReportError::ReductionEscapingRepeatCycle(location, variable, reductionRootLps, false);
	}
	return NULL;
}

// Here we do the dependency analysis twice to take care of any new dependencies that may arise due to the return 
// from the last stage of repeat cycle to the first one. Note that the consequence of this double iteration may 
// seem to be addition of superfluous dependencies when we have nesting of repeat cycles. However that should not 
// happen as there is a redundancy checking mechanism in place where we add dependency arcs to flow stages. So
// unwanted arcs will be dropped off. 
void RepeatControlBlock::performDependencyAnalysis(PartitionHierarchy *hierarchy) {

	// capture any dependency from outside to the execution of the repeat block itself
	FlowStage::performDependencyAnalysis(hierarchy);
	
	// capture the forward dependencies from earlier stages to later stages within the repeat cycle
        CompositeStage::performDependencyAnalysis(hierarchy);
                
	// capture the reverse dependencies later stages to earlier stages within the repeat cycle
	CompositeStage::performDependencyAnalysis(hierarchy);
}

void RepeatControlBlock::analyzeSynchronizationNeeds() {
	FlowStage::analyzeSynchronizationNeeds();
	CompositeStage::analyzeSynchronizationNeeds();
}

//-------------------------------------------------- Condititional Execution Block ----------------------------------------------/

ConditionalExecutionBlock::ConditionalExecutionBlock(Space *space, 
		Expr *executeCond) : CompositeStage(space) {
	this->condition = executeCond;
}

void ConditionalExecutionBlock::performDataAccessChecking(Scope *taskScope) {
        accessMap = validateDataAccess(taskScope, condition, NULL);
        CompositeStage::performDataAccessChecking(taskScope);
}

void ConditionalExecutionBlock::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Conditional Execution: ";
	std::cout << "(Space " << space->getName() << ") ";
	std::cout << "[" << index << "," << groupNo << "," << repeatIndex << "]\n";
        CompositeStage::print(indentLevel);
}

void ConditionalExecutionBlock::calculateLPSUsageStatistics() {
	
	Iterator<VariableAccess*> iterator = accessMap->GetIterator();
        VariableAccess *accessLog;
        
	while ((accessLog = iterator.GetNextValue()) != NULL) {
                if (!accessLog->isContentAccessed()) continue;
                const char *varName = accessLog->getName();
                DataStructure *structure = space->getLocalStructure(varName);
                AccessFlags *accessFlags = accessLog->getContentAccessFlags();
                LPSVarUsageStat *usageStat = structure->getUsageStat();
                if (accessFlags->isRead() || accessFlags->isWritten()) {
                        usageStat->addAccess();
                }
        }
        CompositeStage::calculateLPSUsageStatistics();
}

void ConditionalExecutionBlock::performEpochUsageAnalysis() {
        CompositeStage::performEpochUsageAnalysis();
        FlowStage::CurrentFlowStage = this;
        condition->setEpochVersions(space, 0);
}

void ConditionalExecutionBlock::setLpsExecutionFlags() {
        CompositeStage::setLpsExecutionFlags();
        if (this->isLpsDependent()) {
                space->flagToExecuteCode();
        }
}

void ConditionalExecutionBlock::fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {
        FlowStage::fillInTaskEnvAccessList(envAccessList);
        CompositeStage::fillInTaskEnvAccessList(envAccessList);
}

void ConditionalExecutionBlock::prepareTaskEnvStat(TaskEnvStat *taskStat) {
        FlowStage::prepareTaskEnvStat(taskStat);
        CompositeStage::prepareTaskEnvStat(taskStat);
}

void ConditionalExecutionBlock::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
	FlowStage::performDependencyAnalysis(hierarchy);
        CompositeStage::performDependencyAnalysis(hierarchy);
}

void ConditionalExecutionBlock::analyzeSynchronizationNeeds() {
	FlowStage::analyzeSynchronizationNeeds();
	CompositeStage::analyzeSynchronizationNeeds();
}

//------------------------------------------------------ LPS Transition Block ---------------------------------------------------/

LpsTransitionBlock::LpsTransitionBlock(Space *space, Space *ancestorSpace) : CompositeStage(space) {
	this->ancestorSpace = ancestorSpace;	
}

void LpsTransitionBlock::print(int indentLevel) {
        std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << "Transition to Space " << space->getName() << " { "; 
	std::cout << "[" << index << "," << groupNo << "," << repeatIndex << "]\n";
        CompositeStage::print(indentLevel);
	std::cout << indent.str() << "} // back from Space " << space->getName() << "\n"; 
}

//-----------------------------------------------------  Epoch Boundary Block ---------------------------------------------------/

EpochBoundaryBlock *EpochBoundaryBlock::CurrentEpochBoundary = NULL;

EpochBoundaryBlock::EpochBoundaryBlock(Space *space) : CompositeStage(space) {
	this->lpsToVarMap = new Hashtable<List<const char*>*>;
}

void EpochBoundaryBlock::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Epoch Boundary: ";
	std::cout << "(Space " << space->getName() << ") ";
	std::cout << "[" << index << "," << groupNo << "," << repeatIndex << "]\n";
        CompositeStage::print(indentLevel);
}

void EpochBoundaryBlock::performEpochUsageAnalysis() {
	EpochBoundaryBlock *oldBoundary = EpochBoundaryBlock::CurrentEpochBoundary;
	EpochBoundaryBlock::CurrentEpochBoundary = this;
	CompositeStage::performEpochUsageAnalysis();
	EpochBoundaryBlock::CurrentEpochBoundary = oldBoundary;
}

void EpochBoundaryBlock::recordEpochVariableUsage(const char *varName,  const char *spaceName) {
	List<const char*> *lpsVarList = lpsToVarMap->Lookup(spaceName);
	if (lpsVarList == NULL) {
		lpsVarList = new List<const char*>;
		lpsVarList->Append(varName);
		lpsToVarMap->Enter(spaceName, lpsVarList);	
	} else {
		if (!string_utils::contains(lpsVarList, varName)) {
			lpsVarList->Append(varName);
		}
	}
}

//---------------------------------------------------  Reduction Boundary Block -------------------------------------------------/

ReductionBoundaryBlock::ReductionBoundaryBlock(Space *space) : CompositeStage(space) {
	this->assignedReductions = NULL;
}

void ReductionBoundaryBlock::print(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        std::cout << indent.str() << "Reduction Boundary { ";
        std::cout << "(Space " << space->getName() << ") ";
	std::cout << "[" << index << "," << groupNo << "," << repeatIndex << "]\n";
        CompositeStage::print(indentLevel);
	std::cout << indent.str() << "} // end of reduction " << "\n"; 
}

void ReductionBoundaryBlock::assignReductions(List<ReductionMetadata*> *reductionList) {
	this->assignedReductions = reductionList;
}

void ReductionBoundaryBlock::validateReductions() {
	for (int i = 0; i < assignedReductions->NumElements(); i++) {
		ReductionMetadata *metadata = assignedReductions->Nth(i);
		const char *resultVar = metadata->getResultVar();
		StageInstanciation *executorStage = metadata->getExecutorStage();
		FlowStage *lastAccessor = getLastAccessorStage(resultVar);
		if (executorStage != lastAccessor) {
			ReportError::ReductionVarUsedBeforeReady(lastAccessor->getLocation(), 
				resultVar, space->getName(), false);
		}
	}
	CompositeStage::validateReductions();
}
