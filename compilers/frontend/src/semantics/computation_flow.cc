#include "computation_flow.h"
#include "scope.h"
#include "task_space.h"
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

//------------------------------------------------------ Stage Instanciation ----------------------------------------------------/

StageInstanciation::StageInstanciation(Space *space) : FlowStage(space) {
	this->code = NULL;
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

//------------------------------------------------------- Repeat Control Block --------------------------------------------------/

RepeatControlBlock::RepeatControlBlock(Space *space, 
		RepeatCycleType type, Expr *executeCond) : CompositeStage(space) {
	this->type = type;
	this->condition = executeCond;
}

//-------------------------------------------------- Condititional Execution Block ----------------------------------------------/

ConditionalExecutionBlock::ConditionalExecutionBlock(Space *space, 
		Expr *executeCond) : CompositeStage(space) {
	this->condition = executeCond;
}

//------------------------------------------------------ LPS Transition Block ---------------------------------------------------/

LpsTransitionBlock::LpsTransitionBlock(Space *space, Space *ancestorSpace) : CompositeStage(space) {
	this->ancestorSpace = ancestorSpace;	
}

//-----------------------------------------------------  Epoch Boundary Block ---------------------------------------------------/

EpochBoundaryBlock::EpochBoundaryBlock(Space *space) : CompositeStage(space) {}
