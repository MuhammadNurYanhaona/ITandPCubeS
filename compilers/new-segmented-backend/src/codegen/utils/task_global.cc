#include "task_global.h"

#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/semantics/scope.h"
#include "../../../../frontend/src/semantics/symbol.h"
#include "../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../frontend/src/static-analysis/reduction_info.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cstdlib>	

List<TaskGlobalScalar*> *TaskGlobalCalculator::calculateTaskGlobals(TaskDef *taskDef) {
	
	List<TaskGlobalScalar*> *globalScalars = new List<TaskGlobalScalar*>;
	Scope *scope = taskDef->getSymbol()->getNestedScope();

	List<ReductionMetadata*> *reductionInfos = new List<ReductionMetadata*>;
        taskDef->getComputation()->extractAllReductionInfo(reductionInfos);
	
	// get the name of the repeat loop indexes present in the compute block
	List<const char*> *repeatIndexList = taskDef->getRepeatIndexes();

	Iterator<Symbol*> iterator = scope->get_local_symbols();
	Symbol *symbol = NULL;
	while ((symbol = iterator.GetNextValue()) != NULL) {
		// if the symbol is not a varible then ignore it
		VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
		if (variable == NULL) continue;
		
		Type *type = variable->getType();
		const char *varName = variable->getName();	
		
		// if the variable is an array then it is partitionable and should be ignored
		ArrayType *array = dynamic_cast<ArrayType*>(type);
		if (!(array == NULL)) continue;

		// if the variable is a result variable for some non task-global reduction then it has per
		// LPU instances not a single shared instance for all LPUs
		if (variable->isReduction()) {
			bool singleton = false;
			for (int i = 0; i < reductionInfos->NumElements(); i++) {
				ReductionMetadata *reduction = reductionInfos->Nth(i);
				if (strcmp(reduction->getResultVar(), varName) == 0) {
					singleton = singleton | reduction->isSingleton(); 
				}
			}
			if (!singleton) continue;
		}


		// if the variable is used as a repeat loop index then it is a task-global variable with
		// separate versions for individual PPU controllers
		const char *variableName = variable->getName();
		bool matchFound = false;
		for (int i = 0; i < repeatIndexList->NumElements(); i++) {
			if (strcmp(variableName, repeatIndexList->Nth(i)) == 0) {
				matchFound = true;
				break;
			}
		}

		TaskGlobalScalar *scalar = new TaskGlobalScalar(variableName, matchFound, type);
		globalScalars->Append(scalar);		 
	}
	return globalScalars;
}
