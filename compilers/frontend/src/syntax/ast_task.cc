#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "ast_partition.h"
#include "../common/errors.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../semantics/helper.h"
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
        code = c;
        for (int i = 0; i < code->NumElements(); i++) {
                code->Nth(i)->SetParent(this);
        }
}

void InitializeSection::PrintChildren(int indentLevel) {
        if (arguments->NumElements() > 0) {
                PrintLabel(indentLevel + 1, "Arguments");
                arguments->PrintAll(indentLevel + 2);
        }
        PrintLabel(indentLevel + 1, "Code");
        code->PrintAll(indentLevel + 2);
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

//--------------------------------------------------------------------------------------------------------------------LPS Transition

LpsTransition::LpsTransition(char lpsId, List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {
	this->lpsId = lpsId;
}

void LpsTransition::PrintChildren(int indentLevel) {
	printf(" Space %c", lpsId);
	CompositeFlowPart::PrintChildren(indentLevel);
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

//-----------------------------------------------------------------------------------------------------------------------Epoch Block

EpochBlock::EpochBlock(List<FlowPart*> *nestedSubflow, 
		yyltype loc) : CompositeFlowPart(loc, nestedSubflow) {}

//--------------------------------------------------------------------------------------------------------------------Repeat Control

WhileRepeat::WhileRepeat(Expr *condition, yyltype loc) : RepeatControl(loc) {
	Assert(condition != NULL);
	this->condition = condition;
	this->condition->SetParent(this);
}

void WhileRepeat::PrintChildren(int indentLevel) {
	condition->Print(indentLevel + 1, "Activation-Condition ");
}

ForRepeat::ForRepeat(RangeExpr *rangeExpr, yyltype loc) : RepeatControl(loc) {
	Assert(rangeExpr != NULL);
	this->rangeExpr = rangeExpr;
	this->rangeExpr->SetParent(this);
}

void ForRepeat::PrintChildren(int indentLevel) {
	rangeExpr->Print(indentLevel + 1, "Actition-Condition ");
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

//------------------------------------------------------ Task Definition ----------------------------------------------------------/

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
