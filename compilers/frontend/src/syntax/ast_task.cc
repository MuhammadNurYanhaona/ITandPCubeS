#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "ast_type.h"
#include "ast_task.h"
#include "ast_partition.h"
#include "../common/errors.h"
#include "../common/location.h"
#include "../common/constant.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../semantics/helper.h"
#include "../semantics/computation_flow.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"
#include "../../../common-libs/utils/string_utils.h"

#include <sstream>

//-------------------------------------------------------- Define Section ---------------------------------------------------------/

DefineSection::DefineSection(List<VariableDef*> *def, yyltype loc) : Node(loc) {
        Assert(def != NULL);
        define = def;
        for (int i = 0; i < define->NumElements(); i++) {
                define->Nth(i)->SetParent(this);
        }
}

void DefineSection::PrintChildren(int indentLevel) {
        define->PrintAll(indentLevel + 1);
}

//----------------------------------------------------- Initialize Section --------------------------------------------------------/

InitializeSection::InitializeSection(List<Identifier*> *a, List<Stmt*> *c, yyltype loc) : Node(loc) {
        Assert(a != NULL && c != NULL);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                arguments->Nth(i)->SetParent(this);
        }
        argumentTypes = NULL;
        code = new StmtBlock(c);
	code->SetParent(this);
}

void InitializeSection::PrintChildren(int indentLevel) {
        if (arguments->NumElements() > 0) {
                PrintLabel(indentLevel + 1, "Arguments");
                arguments->PrintAll(indentLevel + 2);
        }
        PrintLabel(indentLevel + 1, "Code");
        code->Print(indentLevel + 2);
}

void InitializeSection::performScopeAndTypeChecking(Scope *parentScope) {
	
	// Generate a parameter scope for initialize arguments
        Scope *parameterScope = new Scope(TaskInitScope);
        TaskDef *taskDef = (TaskDef*) this->parent;
        Scope *taskDefineScope = taskDef->getSymbol()->getNestedScope();
        for (int i = 0; i < arguments->NumElements(); i++) {
                Identifier *id = arguments->Nth(i);
                if (taskDefineScope->lookup(id->getName()) != NULL) {
                        parameterScope->copy_symbol(taskDefineScope->lookup(id->getName()));
                } else {
                        parameterScope->insert_symbol(new VariableSymbol(id->getName(), NULL));
                }
        }

	// enter to the nested scopes for the task and the init section
        Scope *executionScope  = parentScope->enter_scope(taskDefineScope);
        executionScope = executionScope->enter_scope(parameterScope);

        // create a new scope for the init body (code section)
        Scope *initBodyScope = executionScope->enter_scope(new Scope(TaskInitBodyScope));

	// the scope and type analysis should repeat as long as we resolve new expression types
        int iteration = 0;
        int resolvedTypes = 0;
        do {	resolvedTypes = code->resolveExprTypesAndScopes(initBodyScope, iteration);
                iteration++;
        } while (resolvedTypes != 0);

	// emit all scope and type errors, if exist
	code->emitScopeAndTypeErrors(initBodyScope);
	
	// prepare the scopes for storage
	taskDefineScope->detach_from_parent();
        parameterScope->detach_from_parent();
        initBodyScope->detach_from_parent();

        // save parameter and init body scopes
        TaskSymbol *taskSymbol = (TaskSymbol*) taskDef->getSymbol();
        taskSymbol->setInitScope(parameterScope);
        this->scope = initBodyScope;

        // store the argument types for actual to formal parameter matching
        argumentTypes = new List<Type*>;
        for (int i = 0; i < arguments->NumElements(); i++) {
                Identifier *id = arguments->Nth(i);
                VariableSymbol *symbol = (VariableSymbol*) parameterScope->lookup(id->getName());
                if (symbol->getType() == NULL) {
                        ReportError::TypeInferenceError(id, false);
                } else {
                        argumentTypes->Append(symbol->getType());
                }
        }	
}

//----------------------------------------------------- Environment Section -------------------------------------------------------/

//------------------------------------------------------------------------------------------------------------------Environment Link
EnvironmentLink::EnvironmentLink(Identifier *v, LinkageType m) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        var = v;
        var->SetParent(this);
        mode = m;
}

List<EnvironmentLink*> *EnvironmentLink::decomposeLinks(List<Identifier*> *idList, LinkageType mode) {
        List<EnvironmentLink*> *links = new List<EnvironmentLink*>;
        for (int i = 0; i < idList->NumElements(); i++) {
           links->Append(new EnvironmentLink(idList->Nth(i), mode));
        }
        return links;
}

void EnvironmentLink::PrintChildren(int indentLevel) {
        var->Print(indentLevel + 1);
}

const char *EnvironmentLink::GetPrintNameForNode() {
        return (mode == TypeCreate) ? "Create"
                : (mode == TypeLink) ? "Link" : "Create if Not Linked";
}

//---------------------------------------------------------------------------------------------------------------Environment Config
EnvironmentSection::EnvironmentSection(List<EnvironmentLink*> *l, yyltype loc) : Node(loc) {
        Assert(l != NULL);
        links = l;
        for (int i = 0; i < links->NumElements(); i++) {
                links->Nth(i)->SetParent(this);
        }
}

void EnvironmentSection::PrintChildren(int indentLevel) {
        links->PrintAll(indentLevel + 1);
}

//------------------------------------------------------- Stages Section ----------------------------------------------------------/

//------------------------------------------------------------------------------------------------------------------Stage Definition
StageDefinition::StageDefinition(Identifier *name, List<Identifier*> *parameters, Stmt *codeBody) {
	Assert(name != NULL 
		&& parameters != NULL && parameters->NumElements() > 0 
		&& codeBody != NULL);
	this->name = name;
	this->name->SetParent(this);
	this->parameters = parameters;
	for (int i = 0; i < parameters->NumElements(); i++) {
		this->parameters->Nth(i)->SetParent(this);
	}
	this->codeBody = codeBody;
	this->codeBody->SetParent(this);
}

void StageDefinition::PrintChildren(int indentLevel) {
	name->Print(indentLevel + 1, "Name");
        PrintLabel(indentLevel + 1, "Parameters");
        parameters->PrintAll(indentLevel + 2);
        PrintLabel(indentLevel + 1, "Code Body");
	codeBody->Print(indentLevel + 2);
}

void StageDefinition::determineArrayDimensions() {

	Hashtable<semantic_helper::ArrayDimConfig*> *resolvedArrays = new Hashtable<semantic_helper::ArrayDimConfig*>;
	List<Expr*> *arrayAccesses = new List<Expr*>;
	codeBody->retrieveExprByType(arrayAccesses, ARRAY_ACC);

	for (int i = 0; i < arrayAccesses->NumElements(); i++) {
		ArrayAccess *arrayAcc = (ArrayAccess*) arrayAccesses->Nth(i);
		int dimensionality = arrayAcc->getIndexPosition() + 1;
		Expr *baseExpr = arrayAcc->getEndpointOfArrayAccess();
		FieldAccess *baseField = dynamic_cast<FieldAccess*>(baseExpr);

		// This condition should not be true in the general case as arrays cannot hold other arrays 
		// as elements in IT. The only time this will be true is when a static array property of a
		// user defined type instance has been accessed. Those array's dimensions are already known
		// from the type definition.
		if (baseField == NULL || !baseField->isTerminalField()) continue;

		const char *arrayName = baseField->getField()->getName();
		
		// check if the use of the array in the particular expression confirms with any earlier use
		// of the same array
		semantic_helper::ArrayDimConfig	*arrayConfig = NULL;
		if ((arrayConfig = resolvedArrays->Lookup(arrayName)) != NULL) {

			// raise an error that array access dimensions are not matching any previous access
			if (arrayConfig->getDimensions() != dimensionality) {
				ReportError::ConflictingArrayDimensionCounts(arrayAcc->GetLocation(), 
					arrayName, arrayConfig->getDimensions(), dimensionality, false);
			}

		// if this is the first time use of the array then just store it as resolved
		} else {
			arrayConfig = new semantic_helper::ArrayDimConfig(arrayName, dimensionality);
			resolvedArrays->Enter(arrayName, arrayConfig);
		}

		// annotate the field access as accessing an array of a particular dimensionality for later
		// validation against actual invocation arguments
		baseField->flagAsArrayField(dimensionality);			
	}

	delete resolvedArrays;
	delete arrayAccesses;
}

//--------------------------------------------------------------------------------------------------------------------Stages Section

StagesSection::StagesSection(List<StageDefinition*> *stages, yyltype loc) : Node(loc) {
	Assert(stages != NULL && stages->NumElements() > 0);
	this->stages = stages;
}

void StagesSection::PrintChildren(int indentLevel) {
	stages->PrintAll(indentLevel + 1);
}

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

//------------------------------------------------------------------------------------------------------------------Stage Invocation

StageInvocation::StageInvocation(Identifier *stageName, 
		List<Expr*> *arguments, yyltype loc) : FlowPart(loc) {
	Assert(stageName != NULL && arguments != NULL && arguments->NumElements() > 0);
	this->stageName = stageName;
	this->stageName->SetParent(this);
	this->arguments = arguments;
	for (int i = 0; i < arguments->NumElements(); i++) {
		this->arguments->Nth(i)->SetParent(this);
	}
}

void StageInvocation::PrintChildren(int indentLevel) {
	stageName->Print(indentLevel + 1, "Stage ");
	PrintLabel(indentLevel + 1, "Arguments");
	arguments->PrintAll(indentLevel + 2);
}

void StageInvocation::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {
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

void CompositeFlowPart::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {

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

void LpsTransition::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {

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

void ConditionalFlowBlock::constructComputeFlow(CompositeStage *currCompStage,
                        semantic_helper::FlowStageConstrInfo *cnstrInfo) {

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

void EpochBlock::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {
	
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

void RepeatCycle::constructComputeFlow(CompositeStage *currCompStage,
		semantic_helper::FlowStageConstrInfo *cnstrInfo) {
	
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

CompositeStage *ComputationSection::generateComputeFlow(semantic_helper::FlowStageConstrInfo *cnstrInfo) {
	
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

//------------------------------------------------------ Task Definition ----------------------------------------------------------/

TaskDef *TaskDef::currentTask = NULL;

TaskDef::TaskDef(Identifier *i,
                DefineSection *d,
                EnvironmentSection *e,
                InitializeSection *init,
		StagesSection *s,
                ComputationSection *c,
                PartitionSection *p): Definition(*i->GetLocation()) {

        Assert(i != NULL && d != NULL && e != NULL && s != NULL && c != NULL && p != NULL);
        id = i;
        id->SetParent(this);
        define = d;
        define->SetParent(this);
        environment = e;
        environment->SetParent(this);
        initialize = init;
        if (initialize != NULL) {
                initialize->SetParent(this);
        }
	stages = s;
	stages->SetParent(this);
        compute = c;
        compute->SetParent(this);
        partition = p;
        partition->SetParent(this);
}

void TaskDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1);
        define->Print(indentLevel + 1);
        environment->Print(indentLevel + 1);
        if (initialize != NULL) initialize->Print(indentLevel + 1);
	stages->Print(indentLevel + 1);
        compute->Print(indentLevel + 1);
        partition->Print(indentLevel + 1);
}

void TaskDef::analyzeStageDefinitions() {
	List<StageDefinition*> *stageDefs = stages->getStageDefinitions();
	for (int i =0; i < stageDefs->NumElements(); i++) {
		StageDefinition *stageDef = stageDefs->Nth(i);
		stageDef->determineArrayDimensions();
	}
}

List<Type*> *TaskDef::getInitArgTypes() {
	if (initialize == NULL) return new List<Type*>;
        else return initialize->getArgumentTypes();
}

int TaskDef::getPartitionArgsCount() { 
        return partition->getArgumentsCount(); 
}

void TaskDef::attachScope(Scope *parentScope) {

        //--------------------------------create a scope with all the variables declared in the define section
        
	Scope *scope = new Scope(TaskScope);
        List<VariableDef*> *varList = define->getDefinitions();
        for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
                VariableSymbol *varSym = new VariableSymbol(var);
                scope->insert_symbol(varSym);
                if (var->isReduction()) {
                        varSym->flagAsReduction();
                }
        }

        //------------------------------------create the environment scope and at the same time a tuple for it

        List<VariableDef*> *envDef = new List<VariableDef*>;
        List<Type*> *envElementTypes = new List<Type*>;
        Scope *envScope = new Scope(TaskScope);
        List<EnvironmentLink*> *envLinks = environment->getLinks();
        for (int i = 0; i < envLinks->NumElements(); i++) {
                Identifier *var = envLinks->Nth(i)->getVariable();
                Symbol *symbol = scope->lookup(var->getName());
                if (symbol == NULL) {
                        ReportError::UndefinedSymbol(var, false);
                } else {
                        envScope->copy_symbol(symbol);
                        VariableSymbol *varSym = (VariableSymbol*) symbol;
                        envDef->Append(new VariableDef(var, varSym->getType()));
                        envElementTypes->Append(varSym->getType());
                }
        }
        envDef->Append(new VariableDef(new Identifier(*GetLocation(), "name"), Type::stringType));
        envElementTypes->Append(Type::stringType);

        const char *initials = string_utils::getInitials(id->getName());
        char *envTupleName = (char *) malloc(strlen(initials) + 12);
        strcpy(envTupleName, initials);
        strcat(envTupleName, "Environment");

	Identifier *envId = new Identifier(*GetLocation(), envTupleName);
        envTuple = new TupleDef(envId, envDef);
        envTuple->flagAsEnvironment();
        envTuple->setSymbol(new TupleSymbol(envId, envTuple, envElementTypes));
        envTuple->getSymbol()->setNestedScope(envScope);
        parentScope->insert_symbol(envTuple->getSymbol());

	//-------------------------------------------------------create the partition scope and a tuple for it
        
	List<Identifier*> *partitionArgs = partition->getArguments();
        List<VariableDef*> *partitionDef = new List<VariableDef*>;
        List<Type*> *partElementTypes = new List<Type*>;
        char *partTupleName = (char *) malloc(strlen(initials) + 10);
        strcpy(partTupleName, initials);
        strcat(partTupleName, "Partition");
        Identifier *partitionId = new Identifier(*GetLocation(), partTupleName);
        Scope *partScope = new Scope(TaskPartitionScope);
        for (int i = 0; i < partitionArgs->NumElements(); i++) {
                Identifier *arg = partitionArgs->Nth(i);
                VariableDef *var = new VariableDef(arg, Type::intType);
                partitionDef->Append(var);
                partScope->insert_symbol(new VariableSymbol(var));
                partElementTypes->Append(Type::intType);
        }
        partitionTuple = new TupleDef(partitionId, partitionDef);
        partitionTuple->setSymbol(new TupleSymbol(partitionId, partitionTuple, partElementTypes));
        partitionTuple->getSymbol()->setNestedScope(partScope);
        parentScope->insert_symbol(partitionTuple->getSymbol());

        //----------------------------------------------------set the symbol for the task and its nested scope
        
	symbol = new TaskSymbol(id->getName(), this);
        symbol->setNestedScope(scope);
        ((TaskSymbol *) symbol)->setEnvScope(envScope);
        ((TaskSymbol *) symbol)->setPartitionScope(partScope);
        parentScope->insert_symbol(symbol);
}

void TaskDef::typeCheckInitializeSection(Scope *scope) {
	if (initialize != NULL) {
                Scope *executionScope = scope->enter_scope(new Scope(ExecutionScope));
                NamedType *partitionType = new NamedType(partitionTuple->getId());
                executionScope->insert_symbol(new VariableSymbol("partition", partitionType));
                initialize->performScopeAndTypeChecking(executionScope);
        }
}

void TaskDef::constructPartitionHierarchy() {
         partition->constructPartitionHierarchy(this);
}

void TaskDef::constructComputationFlow(Scope *programScope) {

	// set up a static reference of the current task to be accessible during the constrution process
	TaskDef::currentTask = this;

	// retrieve the root of the partition hierarchy
	PartitionHierarchy *lpsHierarchy = partition->getPartitionHierarchy();
	Space *rootLps = lpsHierarchy->getRootSpace();

	// prepare the task scope
	Scope *taskScope = programScope->enter_scope(symbol->getNestedScope());
	Scope *executionScope = taskScope->enter_scope(new Scope(ExecutionScope)); 
	NamedType *partitionType = new NamedType(partitionTuple->getId());
	executionScope->insert_symbol(new VariableSymbol("partition", partitionType));

	// pass control to the Computation Section to prepare the computation flow
	semantic_helper::FlowStageConstrInfo cnstrInfo 
			= semantic_helper::FlowStageConstrInfo(rootLps, executionScope, lpsHierarchy);
	CompositeStage *computation = compute->generateComputeFlow(&cnstrInfo);
}
