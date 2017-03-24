#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../ast_type.h"
#include "../ast_task.h"
#include "../ast_partition.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/computation_flow.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <sstream>

//----------------------------------------------------- Computation Section -------------------------------------------------------/

//-------------------------------------------------------------------------------------------------------------------------Flow Part

int FlowPart::currentFlowIndex = 0;

FlowPart::FlowPart(yyltype loc) : Node(loc) {
	index = currentFlowIndex;
	currentFlowIndex++;
}

void FlowPart::resetFlowIndexRef() {
	currentFlowIndex = 0;
}

//---------------------------------------------------------------------------------------------------------------Composite Flow Part

CompositeFlowPart::CompositeFlowPart(yyltype loc, List<FlowPart*> *nestedSubflow) : FlowPart(loc) {
	Assert(nestedSubflow != NULL && nestedSubflow->NumElements() > 0);
	this->nestedSubflow = nestedSubflow;
	for (int i = 0; i < nestedSubflow->NumElements(); i++) {
		nestedSubflow->Nth(i)->SetParent(this);
	}
}

void CompositeFlowPart::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "NestedFlow");
	nestedSubflow->PrintAll(indentLevel + 2);
}

void CompositeFlowPart::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {

	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();

	for (int i = 0; i < nestedSubflow->NumElements(); i++) {

		// we need to reset this settings as the recursive construction process updates this
		// properties whenever needed
		cnstrInfo->setGroupIndex(group);
		cnstrInfo->setRepeatBlockIndex(repeatBlock);
		cnstrInfo->enterSpace(currLps);

		FlowPart *flowPart = nestedSubflow->Nth(i);
		flowPart->constructComputeFlow(currCompStage, cnstrInfo);
	}
}

//--------------------------------------------------------------------------------------------------------------------LPS Transition

LpsTransition::LpsTransition(char lpsId, List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {
	this->lpsId = lpsId;
}

void LpsTransition::PrintChildren(int indentLevel) {
	printf(" Space %c", lpsId);
	CompositeFlowPart::PrintChildren(indentLevel);
}

void LpsTransition::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {

	Space *lastLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();
	PartitionHierarchy *lpsHierarchy = cnstrInfo->getLpsHierarchy();
	Space *currLps = lpsHierarchy->getSpace(lpsId);

	// ensure that the LPS transition is valid
	if (currLps == NULL) {
		ReportError::SpaceNotFound(GetLocation(), lpsId);
		return;
	} else if (!currLps->isParentSpace(lastLps)) {
		ReportError::InvalidSpaceNesting(GetLocation(), currLps->getName(), lastLps->getName());
		return;
	}

	// set up a new stage and setup the index variables properly
	LpsTransitionBlock *currStage = new LpsTransitionBlock(currLps, lastLps);
	int index = cnstrInfo->getLastStageIndex();
	currStage->setIndex(index);
	currStage->setGroupNo(group);
	currStage->setRepeatIndex(repeatBlock);
	cnstrInfo->advanceLastStageIndex(); 
	cnstrInfo->setGroupIndex(index);
	cnstrInfo->enterSpace(currLps);

	// add the newly created stage to the parent composite stage	
	currCompStage->addStageAtEnd(currStage);

	// process the nested subflow with the newly created flow-stage as the current composite stage
	CompositeFlowPart::constructComputeFlow(currStage, cnstrInfo);
}

//------------------------------------------------------------------------------------------------------------Conditional Flow Block

ConditionalFlowBlock::ConditionalFlowBlock(Expr *conditionExpr, List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {
	Assert(conditionExpr != NULL);
	this->conditionExpr = conditionExpr;
	this->conditionExpr->SetParent(this);
}

void ConditionalFlowBlock::PrintChildren(int indentLevel) {
	conditionExpr->Print(indentLevel + 1, "Condition ");
	CompositeFlowPart::PrintChildren(indentLevel);
}

void ConditionalFlowBlock::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {

	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();
	
	//-------------------------do a scope-and-type validation of the associated condition expression
	Scope *parentScope = cnstrInfo->getScope();
	Scope *executionScope = parentScope->enter_scope(new Scope(ExecutionScope));
	// enter a lpuId variable in the scope if the execution space is partitioned
	Symbol *symbol = currLps->getLpuIdSymbol();
	if (symbol != NULL) { executionScope->insert_symbol(symbol); }
	conditionExpr->resolveExprTypesAndScopes(executionScope);
	int errorCount = conditionExpr->emitScopeAndTypeErrors(executionScope);
	if (errorCount > 0) return;
	if (conditionExpr->getType() != NULL && conditionExpr->getType() != Type::boolType) {
		ReportError::IncompatibleTypes(conditionExpr->GetLocation(), 
				conditionExpr->getType(), Type::boolType, false);
		return;
	}
	//----------------------------------------------------------------end of scope-and-type checking		
	
	// set up a new stage and setup the index variables properly
	ConditionalExecutionBlock *currStage = new ConditionalExecutionBlock(currLps, conditionExpr);
	int index = cnstrInfo->getLastStageIndex();
	currStage->setIndex(index);
	currStage->setGroupNo(group);
	currStage->setRepeatIndex(repeatBlock);
	cnstrInfo->advanceLastStageIndex(); 
	cnstrInfo->setGroupIndex(index);
	
	// add the newly created stage to the parent composite stage	
	currCompStage->addStageAtEnd(currStage);

	// process the nested subflow with the newly created flow-stage as the current composite stage
	CompositeFlowPart::constructComputeFlow(currStage, cnstrInfo);
}

//-----------------------------------------------------------------------------------------------------------------------Epoch Block

EpochBlock::EpochBlock(List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {}

void EpochBlock::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {
	
	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();
	
	// set up a new stage and setup the index variables properly
	EpochBoundaryBlock *currStage = new EpochBoundaryBlock(currLps);
	int index = cnstrInfo->getLastStageIndex();
	currStage->setIndex(index);
	currStage->setGroupNo(group);
	currStage->setRepeatIndex(repeatBlock);
	cnstrInfo->advanceLastStageIndex(); 
	cnstrInfo->setGroupIndex(index);
	
	// add the newly created stage to the parent composite stage	
	currCompStage->addStageAtEnd(currStage);

	// process the nested subflow with the newly created flow-stage as the current composite stage
	CompositeFlowPart::constructComputeFlow(currStage, cnstrInfo);
}

//--------------------------------------------------------------------------------------------------------------------Repeat Control

WhileRepeat::WhileRepeat(Expr *condition, yyltype loc) : RepeatControl(loc) {
	Assert(condition != NULL);
	this->condition = condition;
	this->condition->SetParent(this);
}

void WhileRepeat::PrintChildren(int indentLevel) {
	condition->Print(indentLevel + 1, "Activation-Condition ");
}

int WhileRepeat::validateScopeAndTypes(Scope *executionScope) {
	condition->resolveExprTypesAndScopes(executionScope);
	int errorCount = condition->emitScopeAndTypeErrors(executionScope);
	if (condition->getType() != NULL && condition->getType() != Type::boolType) {
		ReportError::IncompatibleTypes(condition->GetLocation(), 
				condition->getType(), Type::boolType, false);
		errorCount++;
	}
	return errorCount;
}

ForRepeat::ForRepeat(RangeExpr *rangeExpr, yyltype loc) : RepeatControl(loc) {
	Assert(rangeExpr != NULL);
	this->rangeExpr = rangeExpr;
	this->rangeExpr->SetParent(this);
}

void ForRepeat::PrintChildren(int indentLevel) {
	rangeExpr->Print(indentLevel + 1, "Actition-Condition ");
}

int ForRepeat::validateScopeAndTypes(Scope *executionScope) {
	rangeExpr->resolveExprTypesAndScopes(executionScope);
        return rangeExpr->emitScopeAndTypeErrors(executionScope);
}

//----------------------------------------------------------------------------------------------------------------------Repeat Cycle

RepeatCycle::RepeatCycle(RepeatControl *control, List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {
	Assert(control != NULL);
	this->control = control;
	this->control->SetParent(control);
}

void RepeatCycle::PrintChildren(int indentLevel) {
	control->Print(indentLevel + 1);
	CompositeFlowPart::PrintChildren(indentLevel);
}

void RepeatCycle::constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo) {
	
	Space *currLps = cnstrInfo->getCurrSpace();
	int group = cnstrInfo->getCurrGroupIndex();
	int repeatBlock = cnstrInfo->getCurrRepeatBlockIndex();
	
	Expr *condition = control->getRepeatCondition();
	RepeatCycleType cycleType = control->getType();

	// determine the LPS where the repeat block should execute; this is needed as the subpartition
	// repeat is done on the LPS mentioned in the repeat control -- not in the encircling LPS
	Space *repeatLps = currLps;
	SubpartitionRepeat *subpartRepeat = dynamic_cast<SubpartitionRepeat*>(control);
	if (subpartRepeat != NULL) {
		repeatLps = currLps->getSubpartition();
		if (repeatLps == NULL) {
			ReportError::SubpartitionRepeatNotSupported(GetLocation(), currLps->getName());
			return;
		}
		// for subpartition repeat, we can create an LPS transition block as opposed to a 
		// repeat control block as this repeat cycle was there only to set-up the LPS boundary 
		// within which the sub-partitioned LPUs should be generated
		LpsTransitionBlock *currStage = new LpsTransitionBlock(repeatLps, currLps);
		int index = cnstrInfo->getLastStageIndex();
		currStage->setIndex(index);
		currStage->setGroupNo(group);
		currStage->setRepeatIndex(repeatBlock);
		cnstrInfo->advanceLastStageIndex(); 
		cnstrInfo->setGroupIndex(index);
		cnstrInfo->enterSpace(repeatLps);	// flagging the change of the LPS

		// add the newly created stage to the parent composite stage	
		currCompStage->addStageAtEnd(currStage);

		// process the nested subflow with the newly created flow-stage as the current 
		// composite stage
		CompositeFlowPart::constructComputeFlow(currStage, cnstrInfo);
		return;
	}

	//--------------------------------------------------------------------scope-and-type validation
	if (condition != NULL) {
		Scope *parentScope = cnstrInfo->getScope();
		Scope *executionScope = parentScope->enter_scope(new Scope(ExecutionScope));
		// enter a lpuId variable in the scope if the execution space is partitioned
		Symbol *symbol = currLps->getLpuIdSymbol();
		if (symbol != NULL) { executionScope->insert_symbol(symbol); }
		int errorCount = control->validateScopeAndTypes(executionScope);
		if (errorCount > 0) return;
	}
	//-------------------------------------------------------------end of scope and type validation
	
	// set up a new stage and setup the index variables properly
	RepeatControlBlock *currStage = new RepeatControlBlock(repeatLps, cycleType, condition);
	int index = cnstrInfo->getLastStageIndex();
	currStage->setIndex(index);
	currStage->setGroupNo(group);
	currStage->setRepeatIndex(repeatBlock);
	cnstrInfo->advanceLastStageIndex(); 
	cnstrInfo->setGroupIndex(index);
	cnstrInfo->setRepeatBlockIndex(index);	// flagging of the change of the repeat block
	
	// add the newly created stage to the parent composite stage	
	currCompStage->addStageAtEnd(currStage);
	
	// process the nested subflow with the newly created flow-stage as the current composite stage
	CompositeFlowPart::constructComputeFlow(currStage, cnstrInfo);
}

//---------------------------------------------------------------------------------------------------------------Computation Section

ComputationSection::ComputationSection(List<FlowPart*> *cf, yyltype loc) : Node(loc) {
        Assert(cf != NULL && cf->NumElements() > 0);
        computeFlow = cf;
        for (int i = 0; i < computeFlow->NumElements(); i++) {
                computeFlow->Nth(i)->SetParent(this);
        }
}

void ComputationSection::PrintChildren(int indentLevel) {
        computeFlow->PrintAll(indentLevel + 1);
}

CompositeStage *ComputationSection::generateComputeFlow(FlowStageConstrInfo *cnstrInfo) {
	
	Space *currLps = cnstrInfo->getCurrSpace();
	CompositeStage *compStage = new CompositeStage(currLps);

	int index = cnstrInfo->getLastStageIndex();
	int groupIndex = cnstrInfo->getCurrGroupIndex();
	int repeatIndex = cnstrInfo->getCurrRepeatBlockIndex();

	compStage->setIndex(index);
	cnstrInfo->advanceLastStageIndex();
	compStage->setGroupNo(groupIndex - 1);
	compStage->setRepeatIndex(repeatIndex -1);

	for (int i = 0; i < computeFlow->NumElements(); i++) {

		// we need to reset this settings as the recursive construction process updates this
		// properties whenever needed
		cnstrInfo->setGroupIndex(index);
		cnstrInfo->setRepeatBlockIndex(repeatIndex);
		cnstrInfo->enterSpace(currLps);

		FlowPart *flowPart = computeFlow->Nth(i);
		flowPart->constructComputeFlow(compStage, cnstrInfo);
	}

	return compStage;
}

