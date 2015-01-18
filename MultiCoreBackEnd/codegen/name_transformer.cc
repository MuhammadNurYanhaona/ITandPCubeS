#include "name_transformer.h"
#include "../utils/list.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../static-analysis/task_global.h"

#include <string>
#include <iostream>
#include <cstdlib>
#include <sstream>

using namespace ntransform;

NameTransformer *NameTransformer::transformer = NULL;

NameTransformer::NameTransformer() { this->reset(); }

void NameTransformer::reset() {
	taskGlobals = new List<const char*>;
	threadLocals = new List<const char*>;
	globalArrays = new List<const char*>;
	lpuPrefix = "lpu.";
	localAccessDisabled = false;		
}

bool NameTransformer::isTaskGlobal(const char *varName) {
	for (int i = 0; i < taskGlobals->NumElements(); i++) {
		if (strcmp(varName, taskGlobals->Nth(i)) == 0) return true;
	}
	return false;
}

bool NameTransformer::isThreadLocal(const char *varName) {
	for (int i = 0; i < threadLocals->NumElements(); i++) {
		if (strcmp(varName, threadLocals->Nth(i)) == 0) return true;
	}
	return false;
}

bool NameTransformer::isGlobalArray(const char *varName) {
	for (int i = 0; i < globalArrays->NumElements(); i++) {
		if (strcmp(varName, globalArrays->Nth(i)) == 0) return true;
	}
	return false;
}

const char *NameTransformer::getTransformedName(const char *varName, bool metadata, bool local) {
	std::ostringstream xformedName;
	if (isTaskGlobal(varName)) {
		xformedName << "taskGlobals." << varName;
		return strdup(xformedName.str().c_str());
	} else if (isThreadLocal(varName)) {
		xformedName << "threadLocals." << varName;
		return strdup(xformedName.str().c_str());
	} else if (isGlobalArray(varName)) {
		if (metadata) {
			if (local && !localAccessDisabled) {
				xformedName << varName << "PartDims";
				return strdup(xformedName.str().c_str());
			} else {
				xformedName << "arrayMetadata." << varName << "Dims";
				return strdup(xformedName.str().c_str());
			}
		} else {
			xformedName << lpuPrefix << varName;
			return strdup(xformedName.str().c_str());
		}
	}
	return varName;
}

void NameTransformer::setTransformer(TaskDef *taskDef) {
	if (transformer == NULL) {
		transformer = new NameTransformer;
	} else {
		transformer->reset();
	}
	List<TaskGlobalScalar*> *scalarList = TaskGlobalCalculator::calculateTaskGlobals(taskDef);
	for (int i = 0; i < scalarList->NumElements(); i++) {
		TaskGlobalScalar *scalar = scalarList->Nth(i);
		if (scalar->isLocallyManageable()) {
			transformer->threadLocals->Append(scalar->getName());
		} else {
			transformer->taskGlobals->Append(scalar->getName());
		}
	}
	Iterator<Symbol*> iterator = taskDef->getSymbol()->getNestedScope()->get_local_symbols();
	Symbol *symbol;
	while ((symbol = iterator.GetNextValue()) != NULL) {
		const char *varName = symbol->getName();
		if (!(transformer->isTaskGlobal(varName) 
				|| transformer->isThreadLocal(varName))) {
			transformer->globalArrays->Append(varName);
		}
	}
	// default static array parameter used to indentify which LPU is currently been executed
	transformer->globalArrays->Append("lpuId");
}


