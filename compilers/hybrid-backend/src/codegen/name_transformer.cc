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

//-------------------------------------------------------- Name Transformer -----------------------------------------------------/

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
	// This portion is a patch for translating elements within tasks' environment references. We designed
	// the name transformer with only single task in mind. Now because of time shortage we are using it
	// for translating the coordinator program too. Otherwise, we should TODO refactor this logic to do
	// transformation of names properly based on the context.
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

const char *NameTransformer::getArrayIndexStorageSuffix(const char *arrayName,
                                int dimensionNo, int dimensionCount, int indentLevel) {

	std::ostringstream indent;
        for (int i = 0; i < indentLevel + 2; i++) indent << '\t';

	std::ostringstream indexSuffix;
	indexSuffix << " - " << arrayName << "StoreDims[" << dimensionNo << "].range.min";
	indexSuffix << ")";

	bool firstEntry = true;
	for (int i = dimensionCount - 1; i > dimensionNo; i--) {
		if (!firstEntry) {
			indexSuffix << "\n" << indent.str();
		}
		indexSuffix << " * " << arrayName << "StoreDims[" << i << "].length";
		firstEntry = false;
	}

	return strdup(indexSuffix.str().c_str());
}

void NameTransformer::initializePropertyLists(TaskDef *taskDef) {
	
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

//----------------------------------------------------- Hybrid Name Transformer -------------------------------------------------/

HybridNameTransformer::HybridNameTransformer() : NameTransformer() {
	this->gpuMode = false;
	this->applyWarpSuffix = false;
	this->currentLpsName = NULL;
}

const char *HybridNameTransformer::getTransformedName(const char *varName,
		bool metadata, bool local, Type *type) {
	
	if (!gpuMode) {
		return NameTransformer::getTransformedName(varName, metadata, local, type);
	}
	
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
				xformedName << varName << "Space" << currentLpsName << "PRanges";
				if (applyWarpSuffix) {
					xformedName << "[warpId]";
				}
				return strdup(xformedName.str().c_str());
			} else {
				xformedName << "arrayMetadata." << varName << "Dims";
				return strdup(xformedName.str().c_str());
			}
		}
	}
	return varName;
}

const char *HybridNameTransformer::getArrayIndexStorageSuffix(const char *arrayName,        
		int dimensionNo, int dimensionCount, int indentLevel) {
	
	if (!gpuMode) {
		return NameTransformer::getArrayIndexStorageSuffix(arrayName, 
				dimensionNo, dimensionCount, indentLevel);
	}

	std::ostringstream indentStr;
	for (int i = 0; i < indentLevel + 2; i++) indentStr << '\t';

	std::ostringstream indexSuffix;
	indexSuffix << " - ";
	indexSuffix << arrayName << "SRanges[" << dimensionNo << "].range.min";
	indexSuffix << ")";
	for (int i = dimensionCount - 1; i > dimensionNo; i--) {
		indexSuffix << '\n' << indentStr.str();
		indexSuffix << " * (" << arrayName << "SRanges[" << i << "].range.max";
		indexSuffix << " - " << arrayName << "SRanges[" << i << "].range.min + 1)";
	}

	return strdup(indexSuffix.str().c_str());
		
}

//--------------------------------------------------- Transformer Setup Utility -------------------------------------------------/

void ntransform::setTransformer(TaskDef *taskDef,  bool needHybridTransformer) {
	
	NameTransformer *transformer = NameTransformer::transformer;
	if (transformer == NULL) {
		if (needHybridTransformer) {
			transformer = new HybridNameTransformer;
		} else {
			transformer = new NameTransformer;
		}
		NameTransformer::transformer = transformer;
	} else {
		transformer->reset();
	}

	transformer->initializePropertyLists(taskDef);
}


