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
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

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

//-------------------------------------------------- Condititional Execution Block ----------------------------------------------/

ConditionalExecutionBlock::ConditionalExecutionBlock(Space *space, 
		Expr *executeCond) : CompositeStage(space) {
	this->condition = executeCond;
}

void ConditionalExecutionBlock::performDataAccessChecking(Scope *taskScope) {
        accessMap = validateDataAccess(taskScope, condition, NULL);
        CompositeStage::performDataAccessChecking(taskScope);
}

//------------------------------------------------------ LPS Transition Block ---------------------------------------------------/

LpsTransitionBlock::LpsTransitionBlock(Space *space, Space *ancestorSpace) : CompositeStage(space) {
	this->ancestorSpace = ancestorSpace;	
}

//-----------------------------------------------------  Epoch Boundary Block ---------------------------------------------------/

EpochBoundaryBlock::EpochBoundaryBlock(Space *space) : CompositeStage(space) {}
