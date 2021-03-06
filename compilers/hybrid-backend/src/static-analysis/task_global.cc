#include "task_global.h"
#include "data_flow.h"

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"

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

		// if the variable is a result variable for some non task-global reduction then it has per-LPU 
		// instances not a single shared instance for all LPUs
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

		// if the variable of epoch type then it is a locally storable scalar
		if (Type::epochType == type) {
			TaskGlobalScalar *scalar = new TaskGlobalScalar(variable->getName(), 
					true, type);
			globalScalars->Append(scalar);

		// otherwise if it is used as a repeat loop index then again it is a locally stored variable
		} else {
			const char *variableName = variable->getName();
			bool matchFound = false;
			for (int i = 0; i < repeatIndexList->NumElements(); i++) {
				if (strcmp(variableName, repeatIndexList->Nth(i)) == 0) {
					matchFound = true;
					break;
				}
			}
			TaskGlobalScalar *scalar = new TaskGlobalScalar(variableName, 
					matchFound, type);
			globalScalars->Append(scalar);	
		}		 
	}
	return globalScalars;
}
