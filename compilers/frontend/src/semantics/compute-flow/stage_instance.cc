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

#include <iostream>
#include <fstream>
#include <cstdlib>

//------------------------------------------------------ Stage Instanciation ----------------------------------------------------/

StageInstanciation::StageInstanciation(Space *space) : FlowStage(space) {
	this->code = NULL;
	this->nestedReductions = new List<ReductionMetadata*>;
}

void StageInstanciation::performDataAccessChecking(Scope *taskScope) {
	accessMap = validateDataAccess(taskScope, NULL, code);
}

void StageInstanciation::print(int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::cout << indent.str() << "Stage Invocation: " << name << " "; 
	std::cout << "(Space " << space->getName() << ")\n";
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess* accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                accessLog->printAccessDetail(indentLevel + 1);
        }
}

void StageInstanciation::populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress,
		Space *lps, bool includeLimiterLps) {
	if (space->isParentSpace(lps) || (lps == space && includeLimiterLps)) {
		Stmt::mergeAccessedVariables(accessMapInProgress, accessMap);
	}
}

void StageInstanciation::performEpochUsageAnalysis() {
        FlowStage::CurrentFlowStage = this;
        code->analyseEpochDependencies(space);
}

void StageInstanciation::setLpsExecutionFlags() {
        space->flagToExecuteCode();
}

void StageInstanciation::populateReductionMetadata(PartitionHierarchy *lpsHierarchy) {
        code->extractReductionInfo(nestedReductions, lpsHierarchy, space);
	for (int i = 0; i < nestedReductions->NumElements(); i++) {
		ReductionMetadata *metadata = nestedReductions->Nth(i);
		metadata->setExecutorStage(this);
	}
}

void StageInstanciation::extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos) {
        if (nestedReductions != NULL) {
                reductionInfos->AppendAll(nestedReductions);
        }
}

List<ReductionMetadata*> *StageInstanciation::upliftReductionInstrs() {
	return nestedReductions;
}

void StageInstanciation::filterReductionsAtLps(Space *reductionRootLps, 
		List<ReductionMetadata*> *filteredList) {

	List<ReductionMetadata*> *updatedList = new List<ReductionMetadata*>;
	for (int i = 0; i < nestedReductions->NumElements(); i++) {
		
		ReductionMetadata *metadata = nestedReductions->Nth(i);
		Space *currentRootLps = metadata->getReductionRootLps();
		
		if (reductionRootLps == currentRootLps) {
			filteredList->Append(metadata);
		} else {
			updatedList->Append(metadata);
		}
	}
	this->nestedReductions = updatedList;
}

FlowStage *StageInstanciation::getLastAccessorStage(const char *varName) {
	Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                const char *name = accessLog->getName();
		if (strcmp(name, varName) == 0) return this;
	}
	return NULL;
}
