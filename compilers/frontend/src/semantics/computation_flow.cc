#include "computation_flow.h"
#include "scope.h"
#include "symbol.h"
#include "task_space.h"
#include "data_access.h"
#include "../common/errors.h"
#include "../common/location.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

//---------------------------------------------------------- Flow Stage ---------------------------------------------------------/

FlowStage::FlowStage(Space *space) {
        this->space = space;
        this->parent = NULL;
}

Hashtable<VariableAccess*> *FlowStage::validateDataAccess(Scope *taskScope, Expr *activationCond, Stmt *code) {
	
	//------------------------------------------------------------------------first gather the access information

	TaskGlobalReferences *references = new TaskGlobalReferences(taskScope);
	Hashtable<VariableAccess*> *accessMap = new Hashtable<VariableAccess*>;
	if (activationCond != NULL) { 
		accessMap = activationCond->getAccessedGlobalVariables(references);
	}
	if (code != NULL) {
		Hashtable<VariableAccess*> *codeMap = code->getAccessedGlobalVariables(references);
		Stmt::mergeAccessedVariables(accessMap, codeMap);
	}

	//------------------------------------------------------------------------then validate the variable accesses
	
	// partitionable data structures (currently only arrays are partitionable) must be explicitly
        // mentioned in the partition section correspond to the LPS within which a computation stage executes. 
	// Scalar or non partitionable structures, on the other hand, are not specified explicitly. They get 
	// added to the LPS as they have been found to be used within an LPS's flow stages.
        Iterator<VariableAccess*> iter = accessMap->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                const char *name = accessLog->getName();
                if (space->isInSpace(name)) continue;
                VariableSymbol *symbol = (VariableSymbol*) taskScope->lookup(name);
                Type *type = symbol->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                if (arrayType != NULL && staticArray == NULL) {
                        ReportError::ArrayPartitionUnknown(location, name, space->getName());
                } else {
                        DataStructure *source = space->getStructure(name);
                        DataStructure *structure = new DataStructure(source);
                        space->addDataStructure(structure);
                }
        }

	return accessMap;
}

//------------------------------------------------------ Stage Instanciation ----------------------------------------------------/

StageInstanciation::StageInstanciation(Space *space) : FlowStage(space) {
	this->code = NULL;
}

void StageInstanciation::performDataAccessChecking(Scope *taskScope) {
	validateDataAccess(taskScope, NULL, code);
}

//-------------------------------------------------------- Composite Stage ------------------------------------------------------/

CompositeStage::CompositeStage(Space *space) : FlowStage(space) {
	this->stageList = new List<FlowStage*>;
}

void CompositeStage::addStageAtBeginning(FlowStage *stage) {
        stageList->InsertAt(stage, 0);
        stage->setParent(this);
}

void CompositeStage::addStageAtEnd(FlowStage *stage) {
        stageList->Append(stage);
        stage->setParent(this);
}

void CompositeStage::insertStageAt(int index, FlowStage *stage) {
        stageList->InsertAt(stage, index);
        stage->setParent(this);
}

void CompositeStage::removeStageAt(int stageIndex) { stageList->RemoveAt(stageIndex); }

void CompositeStage::performDataAccessChecking(Scope *taskScope) {
	for (int i = 0; i < stageList->NumElements(); i++) {
		FlowStage *stage = stageList->Nth(i);
		stage->performDataAccessChecking(taskScope);
	}
}

//------------------------------------------------------- Repeat Control Block --------------------------------------------------/

RepeatControlBlock::RepeatControlBlock(Space *space, 
		RepeatCycleType type, Expr *executeCond) : CompositeStage(space) {
	this->type = type;
	this->condition = executeCond;
}

void RepeatControlBlock::performDataAccessChecking(Scope *taskScope) {
	validateDataAccess(taskScope, condition, NULL);
	CompositeStage::performDataAccessChecking(taskScope);
}

//-------------------------------------------------- Condititional Execution Block ----------------------------------------------/

ConditionalExecutionBlock::ConditionalExecutionBlock(Space *space, 
		Expr *executeCond) : CompositeStage(space) {
	this->condition = executeCond;
}

void ConditionalExecutionBlock::performDataAccessChecking(Scope *taskScope) {
        validateDataAccess(taskScope, condition, NULL);
        CompositeStage::performDataAccessChecking(taskScope);
}

//------------------------------------------------------ LPS Transition Block ---------------------------------------------------/

LpsTransitionBlock::LpsTransitionBlock(Space *space, Space *ancestorSpace) : CompositeStage(space) {
	this->ancestorSpace = ancestorSpace;	
}

//-----------------------------------------------------  Epoch Boundary Block ---------------------------------------------------/

EpochBoundaryBlock::EpochBoundaryBlock(Space *space) : CompositeStage(space) {}
