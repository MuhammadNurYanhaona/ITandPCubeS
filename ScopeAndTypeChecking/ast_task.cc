#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "ast_partition.h"

#include "errors.h"
#include "list.h"
#include "symbol.h"
#include "scope.h"
#include "task_space.h"

//------------------------------------------- Task ----------------------------------------------/

TaskDef::TaskDef(Identifier *i, DefineSection *d, EnvironmentConfig *e, 
                InitializeInstr *init, ComputeSection *c, PartitionSection *p): Definition(*i->GetLocation()) {
        Assert(i != NULL && d != NULL && e != NULL && c != NULL && p != NULL);
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
	compute = c;
	compute->SetParent(this);
	partition = p;
	partition->SetParent(this);
	envTuple = NULL;
	partitionTuple = NULL;
}

void TaskDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1);
	define->Print(indentLevel + 1);
	environment->Print(indentLevel + 1);
	if (initialize != NULL) initialize->Print(indentLevel + 1);
	compute->Print(indentLevel + 1);
	partition->Print(indentLevel + 1);
}

void TaskDef::attachScope(Scope *parentScope) {
	
	// create a scope with all the variables declared in the define section
	Scope *scope = new Scope(TaskScope);
	List<VariableDef*> *varList = define->getDefinitions();
	for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
                VariableSymbol *varSym = new VariableSymbol(var);
                scope->insert_symbol(varSym);
        }

	// create the environment scope and at the same time a tuple for it
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
	char *envTupleName = (char *) malloc(strlen(id->getName()) + 13);
	strcpy(envTupleName, id->getName());
	strcat(envTupleName, " Environment");
	Identifier *envId = new Identifier(*GetLocation(), envTupleName);
	envTuple = new TupleDef(envId, envDef);
	envTuple->setSymbol(new TupleSymbol(envId, envTuple, envElementTypes));
	envTuple->getSymbol()->setNestedScope(envScope);
	parentScope->insert_symbol(envTuple->getSymbol()); 

	// create the partition scope and a tuple for it
	List<Identifier*> *partitionArgs = partition->getArguments();
	List<VariableDef*> *partitionDef = new List<VariableDef*>;
	List<Type*> *partElementTypes = new List<Type*>;	
	char *partTupleName = (char *) malloc(strlen(id->getName()) + 11);
	strcpy(partTupleName, id->getName());
	strcat(partTupleName, " Partition");
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

	// set the symbol for the task and its nested scope
	symbol = new TaskSymbol(id->getName(), this);
	symbol->setNestedScope(scope);
	((TaskSymbol *) symbol)->setEnvScope(envScope);
	((TaskSymbol *) symbol)->setPartitionScope(partScope);

	// insert task symbol in the parent scope	
	parentScope->insert_symbol(symbol);
}

void TaskDef::validateScope(Scope *parentScope) {
	
	// validate the variables declared in the define section
	List<VariableDef*> *varList = define->getDefinitions();
	for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
		var->validateScope(parentScope);	
	}
	
	// construct and ensure that the task has a valid partition hierarchy
	constructPartitionHierarchy();
	
	// validate the computation logic based on the partition specifications
	validateComputeSection(parentScope);
}

List<Type*> *TaskDef::getInitArgTypes() {
	if (initialize == NULL) return new List<Type*>;
	else return initialize->getArgumentTypes();
}

int TaskDef::getPartitionArgsCount() { 
	return partition->getArgumentsCount(); 
}

void TaskDef::typeCheckInitSection(Scope *scope) {
	if (initialize != NULL) {
		Scope *executionScope = scope->enter_scope(new Scope(ExecutionScope));
		NamedType *partitionType = new NamedType(partitionTuple->getId());
		executionScope->insert_symbol(new VariableSymbol("partition", partitionType));
		initialize->generateScope(executionScope);
	}
}

void TaskDef::constructPartitionHierarchy() {
	 partition->constructPartitionHierarchy(this);
}

void TaskDef::validateComputeSection(Scope *parentScope) {
	
	Scope *defineScope = this->getSymbol()->getNestedScope();	
	defineScope = parentScope->enter_scope(defineScope);

	Scope *executionScope = defineScope->enter_scope(new Scope(ExecutionScope));
	NamedType *partitionType = new NamedType(partitionTuple->getId());
	executionScope->insert_symbol(new VariableSymbol("partition", partitionType));
	
	compute->validateScopes(executionScope, partition->getPartitionHierarchy());
	
	executionScope->exit_scope();
	defineScope->exit_scope();
}

//------------------------------------- Define Section ------------------------------------------/

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

//------------------------------------- Initialize Section ----------------------------------------/

InitializeInstr::InitializeInstr(List<Identifier*> *a, List<Stmt*> *c, yyltype loc) : Node(loc) {
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
	scope = NULL;
}

void InitializeInstr::PrintChildren(int indentLevel) {
	if (arguments->NumElements() > 0) {
		PrintLabel(indentLevel + 1, "Arguments");
		arguments->PrintAll(indentLevel + 2);
	}
	PrintLabel(indentLevel + 1, "Code");
	code->PrintAll(indentLevel + 2);
}

void InitializeInstr::generateScope(Scope *parentScope) {
	
	// Generate a parameter scope for init arguments
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

	// perform semantic checking and type inference
	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);		
		stmt->checkSemantics(initBodyScope, true);
	}

	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);		
		stmt->performTypeInference(initBodyScope);
	}
	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);		
		stmt->performTypeInference(initBodyScope);
	}

	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);		
		stmt->checkSemantics(initBodyScope, false);
	}
	
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

//------------------------------------- Environment Section ----------------------------------------/

//-------------------------------------------------------------------------Environment Link
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

//-----------------------------------------------------------------------Environment Config
EnvironmentConfig::EnvironmentConfig(List<EnvironmentLink*> *l, yyltype loc) : Node(loc) {
	Assert(l != NULL);
	links = l;
	for (int i = 0; i < links->NumElements(); i++) {
		links->Nth(i)->SetParent(this);
	}
}

void EnvironmentConfig::PrintChildren(int indentLevel) {
        links->PrintAll(indentLevel + 1);
}

//------------------------------------- Compute Section ------------------------------------------/

//------------------------------------------------------------------------------Stage Header
StageHeader::StageHeader(Identifier *s, char si, Expr *a) : Node(*s->GetLocation()) {
	Assert(s!= NULL);
	stageId = s;
	stageId->SetParent(this);
	spaceId = si;
	activationCommand = a;
	if (activationCommand != NULL) {
		activationCommand->SetParent(this);
	}
}

void StageHeader::PrintChildren(int indentLevel) {
	stageId->Print(indentLevel + 1);
	PrintLabel(indentLevel + 1, "Space");
	printf("%c", spaceId);
	if (activationCommand != NULL) { 
		PrintLabel(indentLevel + 1, "Activation Command");
		activationCommand->Print(indentLevel + 2); 
	}
}

Space *StageHeader::getExecutionSpace(PartitionHierarchy *partitionHierarchy) {
	
	Space *space = partitionHierarchy->getSubspace(spaceId);
	if (space == NULL) { space = partitionHierarchy->getSpace(spaceId); }
	if (space == NULL) {
		ReportError::SpaceNotFound(GetLocation(), spaceId);
		return partitionHierarchy->getRootSpace();
	}
	return space;
}

void StageHeader::validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {
	if (activationCommand != NULL) {
		activationCommand->resolveType(rootScope, false);
	}
}

//---------------------------------------------------------------------------- Compute Stage
ComputeStage::ComputeStage(StageHeader *h, List<Stmt*> *c) : Node(*h->GetLocation()) {
	
	Assert(h != NULL && c != NULL);
	header = h;
	header->SetParent(this);
	code = c;
	for (int i = 0; i < code->NumElements(); i++) {
		code->Nth(i)->SetParent(this);
	}
	metaStage = false;
	nestedSequence = NULL;
	
	executionSpace = NULL;
	repeatLoopSpace = NULL;
}

ComputeStage::ComputeStage(StageHeader *h, List<MetaComputeStage*> *mcs) : Node(*h->GetLocation()) {
	
	Assert(h != NULL && mcs != NULL);
	header = h;
	header->SetParent(this);
	code = NULL;
	metaStage = true;
	nestedSequence = mcs;
	for (int i = 0; i < nestedSequence->NumElements(); i++) {
		nestedSequence->Nth(i)->SetParent(this);
	}
	
	scope = NULL;
	executionSpaceCap = NULL;	
	executionSpace = NULL;
	repeatLoopSpace = NULL;
}

void ComputeStage::PrintChildren(int indentLevel) {
	header->Print(indentLevel + 1);
	if (metaStage) {
		PrintLabel(indentLevel + 1, "Nested Stages");
		nestedSequence->PrintAll(indentLevel + 2);
	} else {
		PrintLabel(indentLevel + 1, "Code");
		code->PrintAll(indentLevel + 2);
	}
}

Space *ComputeStage::getExecutionSpace(PartitionHierarchy *partitionHierarchy) {
	if (executionSpace == NULL) {
		executionSpace = header->getExecutionSpace(partitionHierarchy);
	}
	return executionSpace;
}

void ComputeStage::validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {
	
	this->executionSpace = header->getExecutionSpace(partitionHierarchy);
	header->validateScope(rootScope, partitionHierarchy);

	if (executionSpace != executionSpaceCap && !executionSpace->isParentSpace(executionSpaceCap)) {
		ReportError::InvalidSpaceNesting(GetLocation(), executionSpace->getName(), executionSpaceCap->getName());	
	}

	Space *subpartitionRoot = executionSpace->getClosestSubpartitionRoot();
	if (subpartitionRoot != NULL 
			&& (repeatLoopSpace == NULL || repeatLoopSpace != subpartitionRoot)) {
		ReportError::SubpartitionRepeatMeesing(GetLocation(), 
				executionSpace->getName(), subpartitionRoot->getName());
	}

	if (metaStage) {
		for (int i = 0; i < nestedSequence->NumElements(); i++) {
			MetaComputeStage *metaStage = nestedSequence->Nth(i);
			metaStage->setExecutionSpace(this->executionSpace);
			metaStage->validateScopes(rootScope, partitionHierarchy);	
		}
	} else {
		// create and attach a scope to the compute stage 
		scope = new Scope(ComputationStageScope);
		rootScope->enter_scope(scope);
			
		// do semantic analysis of the stage body
        	for (int j = 0; j < code->NumElements(); j++) {
                	Stmt *stmt = code->Nth(j);
                	stmt->checkSemantics(scope, true);
        	}
	
        	for (int j = 0; j < code->NumElements(); j++) {
                	Stmt *stmt = code->Nth(j);
                	stmt->performTypeInference(scope);
        	}
        	for (int j = 0; j < code->NumElements(); j++) {
                	Stmt *stmt = code->Nth(j);
                	stmt->performTypeInference(scope);
        	}

        	for (int j = 0; j < code->NumElements(); j++) {
                	Stmt *stmt = code->Nth(j);
                	stmt->checkSemantics(scope, false);
        	}
		scope->exit_scope();
	}
}

//-------------------------------------------------------------------------- Repeat Control
RepeatControl::RepeatControl(Identifier *b, Expr *r, yyltype loc) : Node(loc) {
	Assert(b != NULL && r != NULL);
	begin = b;
	begin->SetParent(this);
	rangeExpr = r;
	rangeExpr->SetParent(this);
	SubpartitionRangeExpr *subPartitionRepeat = dynamic_cast<SubpartitionRangeExpr*>(r);
	subpartitionIteration = (subPartitionRepeat != NULL);
	executionSpace = NULL;
	executionSpaceCap = NULL;
}

void RepeatControl::PrintChildren(int indentLevel) {
	begin->Print(indentLevel + 1, "(GoTo) ");
	rangeExpr->Print(indentLevel + 1, "(If) ");
}

void RepeatControl::validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {
	rangeExpr->resolveType(rootScope, false);
}

Space *RepeatControl::getExecutionSpace(PartitionHierarchy *partitionHierarchy) {
	if (executionSpace != NULL) return executionSpace;
	if (subpartitionIteration) {
		char spaceId = ((SubpartitionRangeExpr*) rangeExpr)->getSpaceId();
		Space *space = partitionHierarchy->getSubspace(spaceId);
		if (space == NULL) {
			if (partitionHierarchy->getSpace(spaceId) != NULL) {
				ReportError::SubpartitionRepeatNotSupported(GetLocation(), spaceId);
			} else {
				ReportError::SpaceNotFound(GetLocation(), spaceId);
			}
			executionSpace = partitionHierarchy->getRootSpace();
		} else {
			executionSpace = space;
		}
	} else {
		executionSpace = partitionHierarchy->getRootSpace();
	}
	return executionSpace;
}

void RepeatControl::setExecutionSpaceCap(Space *space, PartitionHierarchy *partitionHierarchy) {
	if (subpartitionIteration) {
		getExecutionSpace(partitionHierarchy);
		if (executionSpace != space && !executionSpace->isParentSpace(space)) {
			ReportError::ImpermissibleRepeat(GetLocation(), 
					space->getName(), executionSpace->getName());	
		}
		executionSpaceCap = executionSpace;
	} else {
		executionSpaceCap = space;
	}
}

void RepeatControl::setExecutionSpace(Space *space, PartitionHierarchy *partitionHierarchy) {
	if (subpartitionIteration) {
		getExecutionSpace(partitionHierarchy);
		if (!space->isParentSpace(executionSpace)) {
			ReportError::RepeatLoopAdvanceImposssible(GetLocation(), 
					space->getName(), executionSpace->getName());
		}
	} else {
		if (executionSpaceCap != space && !space->isParentSpace(executionSpaceCap)) {
			ReportError::RepeatLoopAdvanceImposssible(GetLocation(), 
					space->getName(), executionSpaceCap->getName());
		}	
		executionSpace = space;
	}
}

//-------------------------------------------------------------------------- Stage Sequence
MetaComputeStage::MetaComputeStage(List<ComputeStage*> *s, RepeatControl *r) : Node() {
	Assert(s != NULL);
	stageSequence = s;
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		stageSequence->Nth(i)->SetParent(this);
	}
	repeatInstr = r;
	if (repeatInstr != NULL) {
		repeatInstr->SetParent(this);
	}
}

void MetaComputeStage::PrintChildren(int indentLevel) {
	stageSequence->PrintAll(indentLevel + 1);
	if (repeatInstr != NULL) repeatInstr->Print(indentLevel + 1);
}

void MetaComputeStage::validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {
	
	if (repeatInstr != NULL) {
		repeatInstr->setExecutionSpaceCap(this->executionSpace, partitionHierarchy);
		repeatInstr->validateScopes(rootScope, partitionHierarchy);
		const char *repeatBegin = repeatInstr->getFirstComputeStageName();
		int startStageIndex = 0;
		bool stageFound = false; 
		for (int i = 0; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			if (strcmp(stage->getName(), repeatBegin) == 0) {
				stageFound = true;
				startStageIndex = i;
				break;
			}
		}
		if (!stageFound) {
			ReportError::RepeatBeginningInvalid(repeatInstr->GetLocation(), 
					stageSequence->Nth(0)->getName());
		} else {
			Space *topSpaceInSeq = stageSequence->Nth(
						startStageIndex)->getExecutionSpace(partitionHierarchy);
			for (int i = startStageIndex + 1; i < stageSequence->NumElements(); i++) {
				ComputeStage *stage = stageSequence->Nth(i);
				topSpaceInSeq = partitionHierarchy->getCommonAncestor(topSpaceInSeq, 
							stage->getExecutionSpace(partitionHierarchy));
			}
			repeatInstr->setExecutionSpace(topSpaceInSeq, partitionHierarchy);
			Space *repeatSpace = repeatInstr->getExecutionSpace(partitionHierarchy);
			for (int i = startStageIndex; i < stageSequence->NumElements(); i++) {
				stageSequence->Nth(i)->setRepeatLoopSpace(repeatSpace);
			}
		}
	} 
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		ComputeStage *stage = stageSequence->Nth(i);
		stage->setExecutionSpaceCap(this->executionSpace);
		stage->validateScope(rootScope, partitionHierarchy);
		Space *stageSpace = stage->getExecutionSpace(partitionHierarchy);
	}
}

//------------------------------------------------------------------------- Compute Section
ComputeSection::ComputeSection(List<MetaComputeStage*> *s, yyltype loc) : Node(loc) {
	Assert(s != NULL);
	stageSeqList = s;
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		stageSeqList->Nth(i)->SetParent(this);
	}
}

void ComputeSection::PrintChildren(int indentLevel) {
	stageSeqList->PrintAll(indentLevel + 1);
}

void ComputeSection::validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {	
	Space *rootSpace = partitionHierarchy->getRootSpace();
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		MetaComputeStage *metaStage = stageSeqList->Nth(i);
		metaStage->setExecutionSpace(rootSpace);
		metaStage->validateScopes(rootScope, partitionHierarchy);	
	}
}


