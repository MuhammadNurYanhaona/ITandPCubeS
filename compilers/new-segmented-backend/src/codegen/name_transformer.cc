#include "name_transformer.h"
#include "task_global.h"

#include "../../../common-libs/utils/list.h"

#include "../../../frontend/src/syntax/ast_task.h"
#include "../../../frontend/src/syntax/ast_def.h"
#include "../../../frontend/src/syntax/ast_type.h"
#include "../../../frontend/src/semantics/scope.h"
#include "../../../frontend/src/semantics/symbol.h"

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
	lpuPrefix = "lpu->";
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

const char *NameTransformer::getTransformedName(const char *varName, bool metadata, bool local, Type *type) {
	std::ostringstream xformedName;
	if (isTaskGlobal(varName)) {
		xformedName << "taskGlobals->" << varName;
		return strdup(xformedName.str().c_str());
	} else if (isThreadLocal(varName)) {
		xformedName << "threadLocals->" << varName;
		return strdup(xformedName.str().c_str());
	} else if (isGlobalArray(varName)) {
		if (metadata) {
			if (local && !localAccessDisabled) {
				xformedName << varName << "PartDims";
				return strdup(xformedName.str().c_str());
			} else {
				xformedName << "arrayMetadata->" << varName << "Dims";
				return strdup(xformedName.str().c_str());
			}
		} else {
			xformedName << lpuPrefix << varName;
			return strdup(xformedName.str().c_str());
		}
	}
	//------------------------------------------------------------------------------------ Patch Solution
	// This portion is a patch for translating elements within tasks' environment references. We 
	// originally designed the name transformer with only single task in mind. Now because of time 
	// shortage we are using it for translating names in the the program coordinator function too. 
	// TODO Otherwise, we should refactor this logic to do transformation of names properly based 
	// on the context.
	if (type != NULL) {
		ArrayType *array = dynamic_cast<ArrayType*>(type);
		if (array != NULL && metadata) {
			xformedName << varName << "Dims";
			return strdup(xformedName.str().c_str());
		}
	}
	//---------------------------------------------------------------------------------------------------
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


