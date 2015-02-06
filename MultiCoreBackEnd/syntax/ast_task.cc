#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "ast_partition.h"

#include "errors.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../semantics/symbol.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../static-analysis/data_flow.h"
#include "../static-analysis/data_access.h"

#include <sstream>

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

	const char *initials = string_utils::getInitials(id->getName());
	char *envTupleName = (char *) malloc(strlen(initials) + 12);
	strcpy(envTupleName, initials);
	strcat(envTupleName, "Environment");
	
	
	Identifier *envId = new Identifier(*GetLocation(), envTupleName);
	envTuple = new TupleDef(envId, envDef);
	envTuple->setSymbol(new TupleSymbol(envId, envTuple, envElementTypes));
	envTuple->getSymbol()->setNestedScope(envScope);
	parentScope->insert_symbol(envTuple->getSymbol()); 

	// create the partition scope and a tuple for it
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

	// set the symbol for the task and its nested scope
	symbol = new TaskSymbol(id->getName(), this);
	symbol->setNestedScope(scope);
	((TaskSymbol *) symbol)->setEnvScope(envScope);
	((TaskSymbol *) symbol)->setPartitionScope(partScope);

	// insert task symbol in the parent scope	
	parentScope->insert_symbol(symbol);
}

void TaskDef::validateScope(Scope *parentScope) {
	
	// First the define section needs to be validated and all defined variables need to be put into
	// the task scope so that further analysis of other sections can be done
	List<VariableDef*> *varList = define->getDefinitions();
	for (int i = 0; i < varList->NumElements(); i++) {
                VariableDef *var = varList->Nth(i);
		var->validateScope(parentScope);	
	}
	
	// Partition section is validated and a partition hierarchy is constracted before we investigate
	// the compute section as what is valid for compute section in different stages (i.e., its compute
	// stages) is dictated by the partition configuration. For example, if a variable does not exist
	// in an LPS then it cannot be used within any compute stage executing in that LPS.
	constructPartitionHierarchy();
	
	// Finally, validate the computation logic based on the partition specifications.
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

// Note that the ordering of the function calls within the whole analysis procedure is important as 
// there are internal depdendency among these analyses through their input/output data structures. 
void TaskDef::analyseCode() {
	Scope *scope = symbol->getNestedScope();
	// determine what task global variable is been used where to prepare for dependency analysis
	initialize->performVariableAccessAnalysis(scope);
	compute->performVariableAccessAnalysis(scope);
	// convert the body of the compute section into a flow definition for execution control and 
	// data movements	
	PartitionHierarchy *hierarchy = partition->getPartitionHierarchy();
	compute->constructComputationFlow(hierarchy->getRootSpace());
	// determine the read-write dependencies that occur as flow of computation moves along stages	
	compute->performDependencyAnalysis(hierarchy);
	// assign stages stage, group, and nesting indexes to aid latter analysis
	compute->getComputation()->assignIndexAndGroupNo(0, 0, 0);
	// determine what dependency relationships should be translated into synchronization require-
	// ments and recursively mark the sources of these synchronization signals	
	compute->getComputation()->analyzeSynchronizationNeeds();
	// similar to the above determine the sync stages for all synchronization signals
	compute->getComputation()->deriveSynchronizationDependencies();
	// then lift up all sync sources to their proper upper level composite stages
	compute->getComputation()->analyzeSynchronizationNeedsForComposites();
	// finally determine which sync stage for a synchronization  will reset the synchronization
	// primitives so that they can be reused (e.g., for latter iterations)
	compute->getComputation()->setReactivatorFlagsForSyncReqs();
}

void TaskDef::print() {
	printf("------------------------------------------------------------------\n");
	printf("Task: %s", id->getName());
	printf("\n------------------------------------------------------------------\n");
	initialize->printUsageStatistics();
	compute->print();
}

PartitionHierarchy *TaskDef::getPartitionHierarchy() { 
	return partition->getPartitionHierarchy(); 
}

List<Identifier*> *TaskDef::getPartitionArguments() { 
	return partition->getArguments(); 
}

CompositeStage *TaskDef::getComputation() { return compute->getComputation(); }

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
	accessMap = NULL;
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

void InitializeInstr::performVariableAccessAnalysis(Scope *taskGlobalScope) {
	accessMap = new Hashtable<VariableAccess*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Identifier *id = arguments->Nth(i);
		if (taskGlobalScope->lookup(id->getName()) != NULL) {
			VariableAccess *accessLog = new VariableAccess(id->getName());
			accessLog->markContentAccess();
			accessLog->getContentAccessFlags()->flagAsWritten();
			accessMap->Enter(id->getName(), accessLog, true);
		}
	}
	TaskGlobalReferences *references = new TaskGlobalReferences(taskGlobalScope);
	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);		
		Hashtable<VariableAccess*> *table = stmt->getAccessedGlobalVariables(references);
		Stmt::mergeAccessedVariables(accessMap, table);
	}
}

void InitializeInstr::printUsageStatistics() {
	printf("Initialization Section:\n");
	Iterator<VariableAccess*> iter = accessMap->GetIterator();
	VariableAccess* accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		accessLog->printAccessDetail(1);
	}	
}

List<const char*> *InitializeInstr::getArguments() {
	List<const char*> *argNameList = new List<const char*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Identifier *id = arguments->Nth(i);
		argNameList->Append(id->getName());
	}
	return argNameList;
}

void InitializeInstr::generateCode(std::ostringstream &stream) {

	// declare all local variables found in the scope
	Iterator<Symbol*> iterator = scope->get_local_symbols();
	Symbol *symbol;
	while ((symbol = iterator.GetNextValue()) != NULL) {
		VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
		if (variable == NULL) continue;
		Type *type = variable->getType();
		const char *name = variable->getName();
		stream << "\t" << type->getCppDeclaration(name) << ";\n";
	}

	TaskDef *taskDef = (TaskDef*) this->parent;
	Space *space = taskDef->getPartitionHierarchy()->getRootSpace();

	// translate statements into C++ code
	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);
		stmt->generateCode(stream, 1, space);
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

//--------------------------------------------------------------- Deprecated Data Flow Stage
void DataFlowStage::setNestingIndex(int nestingIndex) { this->nestingIndex = nestingIndex; }
void DataFlowStage::setNestingController(DataFlowStage *con) { nestingController = con; }
int DataFlowStage::getNestingIndex() { return nestingIndex; }
void DataFlowStage::setComputeIndex(int computeIndex) { this->computeIndex = computeIndex; }
int DataFlowStage::getComputeIndex() { return computeIndex; }
Hashtable<VariableAccess*> *DataFlowStage::getAccessMap() { return accessMap; }
Space *DataFlowStage::getSpace() { return executionSpace; }

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

void StageHeader::checkVariableAccess(Scope *taskGlobalScope, Hashtable<VariableAccess*> *accessMap) {
	if (activationCommand == NULL) return;
	TaskGlobalReferences *references = new TaskGlobalReferences(taskGlobalScope);
	Hashtable<VariableAccess*> *table = activationCommand->getAccessedGlobalVariables(references);
	Stmt::mergeAccessedVariables(accessMap, table);
}

//---------------------------------------------------------------------------- Compute Stage
ComputeStage::ComputeStage(StageHeader *h, List<Stmt*> *c) : DataFlowStage(*h->GetLocation()) {
	
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
	accessMap = NULL;
	dataDependencies = new DataDependencies();
}

ComputeStage::ComputeStage(StageHeader *h, List<MetaComputeStage*> *mcs) : DataFlowStage(*h->GetLocation()) {
	
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
	accessMap = NULL;
	dataDependencies = new DataDependencies();
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
		this->scope = new Scope(ComputationStageScope);
		rootScope->enter_scope(scope);

		// enter a lpuId variable in the scope if the execution space is partitioned
		Symbol *symbol = executionSpace->getLpuIdSymbol();
		if (symbol != NULL) {
			scope->insert_symbol(symbol);
		}
			
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

		// remove the lpuId variable if exists
		if (symbol != NULL) {
			scope->remove_symbol(symbol->getName());
		}
	
		scope->exit_scope();
	}
}

void ComputeStage::setRepeatLoopSpace(Space *space) {
	this->repeatLoopSpace = space;
	if (metaStage) {
		for (int i = 0; i < nestedSequence->NumElements(); i++) {
			MetaComputeStage *nestedMetaStage = nestedSequence->Nth(i);
			nestedMetaStage->setRepeatLoopSpace(space);
		}
	}
}

void ComputeStage::checkVariableAccess(Scope *taskGlobalScope) {
	accessMap = new Hashtable<VariableAccess*>;
	header->checkVariableAccess(taskGlobalScope, accessMap);

	if (!metaStage) {
		TaskGlobalReferences *references = new TaskGlobalReferences(taskGlobalScope); 
        	for (int j = 0; j < code->NumElements(); j++) {
                	Stmt *stmt = code->Nth(j);
                	Hashtable<VariableAccess*> *table = stmt->getAccessedGlobalVariables(references);
			Stmt::mergeAccessedVariables(accessMap, table);
        	}
	} else {
		for (int i = 0; i < nestedSequence->NumElements(); i++) {
			MetaComputeStage *nestedMetaStage = nestedSequence->Nth(i);
			nestedMetaStage->checkVariableAccess(taskGlobalScope);
		}
	}

	Iterator<VariableAccess*> iter = accessMap->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		const char *name = accessLog->getName();
		if (executionSpace->isInSpace(name)) continue;
		VariableSymbol *symbol = (VariableSymbol*) taskGlobalScope->lookup(name);
		Type *type = symbol->getType();
		ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
		if (arrayType != NULL) {
			ReportError::ArrayPartitionUnknown(GetLocation(), 
					name, header->getStageName(), header->getSpaceId());
		} else {
			DataStructure *source = executionSpace->getStructure(name);
			DataStructure *structure = new DataStructure(source);
			executionSpace->addDataStructure(structure);
		}
	}
}

int ComputeStage::assignFlowStageAndNestingIndexes(int currentNestingIndex, 
		int currentStageIndex, List<DataFlowStage*> *currentStageList) {
	if (metaStage) {
		for (int i = 0; i < nestedSequence->NumElements(); i++) {
			MetaComputeStage *metaStage = nestedSequence->Nth(i);
			currentStageIndex = metaStage->assignFlowStageAndNestingIndexes(
					currentNestingIndex, currentStageIndex, currentStageList);
		}
		return currentStageIndex;
	}
	this->setNestingIndex(currentNestingIndex);
	this->setComputeIndex(currentStageIndex);
	currentStageList->Append(this);
	return currentStageIndex + 1;
}

void ComputeStage::constructComputationFlow(List<FlowStage*> *inProgressStageList, CompositeStage *currentContainerStage) {
	Expr *executeCond = header->getActivationCommand();
	Space *space = this->getSpace();
	int index = inProgressStageList->NumElements();
	FlowStage *flowStage = NULL;	
	if (!metaStage) {
		ExecutionStage *stage = new ExecutionStage(index, space, executeCond);
		inProgressStageList->Append(stage);
		stage->setCode(code);
		stage->setScope(scope);
		flowStage = stage;		
        	currentContainerStage->addSyncStagesBeforeExecution(flowStage, inProgressStageList);
	} else {
		CompositeStage *stage = new CompositeStage(index, space, executeCond);
		inProgressStageList->Append(stage);
        	currentContainerStage->addSyncStagesBeforeExecution(stage, inProgressStageList);
		for (int i = 0; i < nestedSequence->NumElements(); i++) {
			MetaComputeStage *metaStage = nestedSequence->Nth(i);
			metaStage->constructComputationFlow(inProgressStageList, stage);
		}
		stage->addSyncStagesOnReturn(inProgressStageList);
		flowStage = stage;
	}
	flowStage->setName(header->getStageName());
	flowStage->setAccessMap(this->getAccessMap());
        currentContainerStage->addStageAtEnd(flowStage);
}

Hashtable<VariableAccess*> *ComputeStage::getAccessMap() {
        if (!metaStage) return accessMap;
        Hashtable<VariableAccess*> *aggregateMap = new Hashtable<VariableAccess*>;
        for (int i = 0; i < nestedSequence->NumElements(); i++) {
                MetaComputeStage *nestedMetaStage = nestedSequence->Nth(i);
                Hashtable<VariableAccess*> *nestedTable = nestedMetaStage->getAggregateAccessMapOfNestedStages();
                Stmt::mergeAccessedVariables(aggregateMap, nestedTable);
        }
        return aggregateMap;
}

void ComputeStage::populateRepeatIndexes(List <const char*> *currentList) {
	if (metaStage) {
        	for (int i = 0; i < nestedSequence->NumElements(); i++) {
                	MetaComputeStage *nestedMetaStage = nestedSequence->Nth(i);
			nestedMetaStage->populateRepeatIndexes(currentList);
		}
	}
}

//-------------------------------------------------------------------------- Repeat Control
RepeatControl::RepeatControl(Identifier *b, Expr *r, yyltype loc) : DataFlowStage(loc) {
	Assert(b != NULL && r != NULL);
	begin = b;
	begin->SetParent(this);
	rangeExpr = r;
	rangeExpr->SetParent(this);
	SubpartitionRangeExpr *subPartitionRepeat = dynamic_cast<SubpartitionRangeExpr*>(r);
	subpartitionIteration = (subPartitionRepeat != NULL);
	executionSpace = NULL;
	executionSpaceCap = NULL;
	accessMap = new Hashtable<VariableAccess*>;
	traversed = false;
	dataDependencies = new DataDependencies();
}

void RepeatControl::PrintChildren(int indentLevel) {
	begin->Print(indentLevel + 1, "(GoTo) ");
	rangeExpr->Print(indentLevel + 1, "(If) ");
}

void RepeatControl::validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy) {
	// enter a lpuId variable in the scope if the execution space is partitioned
	Symbol *symbol = NULL;
	if (executionSpace != NULL) symbol = executionSpace->getLpuIdSymbol();
	if (symbol != NULL) {
		rootScope->insert_symbol(symbol);
	}
	rangeExpr->resolveType(rootScope, executionSpace == NULL);
	// remove the lpuId variable if exists
	if (symbol != NULL) {
		rootScope->remove_symbol(symbol->getName());
	}
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
		if (space != executionSpace && !space->isParentSpace(executionSpace)) {
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

void RepeatControl::checkVariableAccess(Scope *taskGlobalScope) {
	
	TaskGlobalReferences *references = new TaskGlobalReferences(taskGlobalScope);
	accessMap = rangeExpr->getAccessedGlobalVariables(references);

	Iterator<VariableAccess*> iter = accessMap->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		
		const char *name = accessLog->getName();
		if (executionSpace->isInSpace(name)) continue;
		if (!accessLog->isContentAccessed()) continue;
	
		VariableSymbol *symbol = (VariableSymbol*) taskGlobalScope->lookup(name);
		Type *type = symbol->getType();
		ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
		if (arrayType != NULL) {
			ReportError::ArrayPartitionUnknown(GetLocation(), 
					name, "Repeat Loop", executionSpace->getName()[0]);
		} else {
			DataStructure *source = executionSpace->getStructure(name);
			DataStructure *structure = new DataStructure(source);
			executionSpace->addDataStructure(structure);
		}
	}
}

void RepeatControl::populateRepeatIndexes(List <const char*> *currentList) {
	if (!subpartitionIteration) {
		RangeExpr *expr = dynamic_cast<RangeExpr*>(rangeExpr);
		if (expr != NULL) {
			currentList->Append(expr->getIndexName());
		}
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
			repeatInstr->validateScopes(rootScope, partitionHierarchy);
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


void MetaComputeStage::setRepeatLoopSpace(Space *space) {
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		ComputeStage *stage = stageSequence->Nth(i);
		stage->setRepeatLoopSpace(space);
	}
}

void MetaComputeStage::checkVariableAccess(Scope *taskGlobalScope) {
	
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		ComputeStage *stage = stageSequence->Nth(i);
		stage->checkVariableAccess(taskGlobalScope);	
	}
	if (repeatInstr != NULL) { 
		repeatInstr->checkVariableAccess(taskGlobalScope);
	}
}

int MetaComputeStage::assignFlowStageAndNestingIndexes(int currentNestingIndex, 
			int currentStageIndex, List<DataFlowStage*> *currentStageList) {
	if (repeatInstr == NULL) {
		for (int i = 0; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			currentStageIndex = stage->assignFlowStageAndNestingIndexes(
					currentNestingIndex, currentStageIndex, currentStageList);
		}
		return currentStageIndex;
	} else {
		const char *repeatBegin = repeatInstr->getFirstComputeStageName();
		repeatInstr->setNestingIndex(currentNestingIndex);
		int i = 0;
		for (; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			if (strcmp(stage->getName(), repeatBegin) == 0) break;
			currentStageIndex = stage->assignFlowStageAndNestingIndexes(
					currentNestingIndex, currentStageIndex, currentStageList);
		}
		currentNestingIndex++;
		for (; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			currentStageIndex = stage->assignFlowStageAndNestingIndexes(
					currentNestingIndex, currentStageIndex, currentStageList);
			stage->setNestingController(repeatInstr);
		}
		repeatInstr->setComputeIndex(currentStageIndex);
		currentStageList->Append(repeatInstr);
		return currentStageIndex + 1;
	}
}

Hashtable<VariableAccess*> *MetaComputeStage::getAggregateAccessMapOfNestedStages() {
	Hashtable<VariableAccess*> *accessMap = new Hashtable<VariableAccess*>;
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		ComputeStage *stage = stageSequence->Nth(i);
		Stmt::mergeAccessedVariables(accessMap, stage->getAccessMap());	
	}
	if (repeatInstr != NULL) {
		Stmt::mergeAccessedVariables(accessMap, repeatInstr->getAccessMap());
	}
	return accessMap;	
}

void MetaComputeStage::constructComputationFlow(List<FlowStage*> *inProgressStageList, CompositeStage *currentContainerStage) {
	if (repeatInstr == NULL) {
		for (int i = 0; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			stage->constructComputationFlow(inProgressStageList, currentContainerStage);
		}
	} else {
		const char *repeatBegin = repeatInstr->getFirstComputeStageName();
		int i = 0;
		for (; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			if (strcmp(stage->getName(), repeatBegin) == 0) break;
			stage->constructComputationFlow(inProgressStageList, currentContainerStage);
		}
		Space *repeatSpace = repeatInstr->getSpace();
		RepeatCycleType cycleType = Conditional_Repeat;
		if (repeatSpace->isSubpartitionSpace()) {
			repeatSpace = repeatSpace->getParent();
			cycleType = Subpartition_Repeat;
		}
		
		int index = inProgressStageList->NumElements();
		Expr *repeatCondition = repeatInstr->getCondition();
		RepeatCycle *repeatCycle = new RepeatCycle(index, repeatSpace, cycleType, repeatCondition);
		inProgressStageList->Append(repeatCycle);
		currentContainerStage->addSyncStagesBeforeExecution(repeatCycle, inProgressStageList);
		
		Hashtable<VariableAccess*> *accessLogs = new Hashtable<VariableAccess*>;
		Stmt::mergeAccessedVariables(accessLogs, repeatInstr->getAccessMap());
		for (; i < stageSequence->NumElements(); i++) {
			ComputeStage *stage = stageSequence->Nth(i);
			stage->constructComputationFlow(inProgressStageList, repeatCycle);
			Stmt::mergeAccessedVariables(accessLogs, stage->getAccessMap());	
		}
		repeatCycle->setAccessMap(accessLogs);
		repeatCycle->setRepeatConditionAccessMap(repeatInstr->getAccessMap());
		repeatCycle->setName("\"Repeat Cycle\"");

		repeatCycle->addSyncStagesOnReturn(inProgressStageList);		
		currentContainerStage->addStageAtEnd(repeatCycle);			
	}
}

void MetaComputeStage::populateRepeatIndexes(List <const char*> *currentList) {
	for (int i = 0; i < stageSequence->NumElements(); i++) {
		ComputeStage *stage = stageSequence->Nth(i);
		stage->populateRepeatIndexes(currentList);
	}
	if (repeatInstr != NULL) {
		repeatInstr->populateRepeatIndexes(currentList);
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

void ComputeSection::performVariableAccessAnalysis(Scope *taskGlobalScope) {
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		MetaComputeStage *metaStage = stageSeqList->Nth(i);
		metaStage->checkVariableAccess(taskGlobalScope);
	}
}

void ComputeSection::assignFlowStageAndNestingIndexes(List<DataFlowStage*> *currentStageList) {
	int startingStageIndex = 1;
	int startingNestingIndex = 1;
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		MetaComputeStage *metaStage = stageSeqList->Nth(i);
		startingStageIndex = metaStage->assignFlowStageAndNestingIndexes(
				startingNestingIndex, startingStageIndex, currentStageList);
	}
}

void ComputeSection::constructComputationFlow(Space *rootSpace) {
	CompositeStage *containerStage = new CompositeStage(0, rootSpace, NULL);
	List<FlowStage*> *currentStageList = new List<FlowStage*>;
	currentStageList->Append(containerStage);
	SpaceEntryCheckpoint::addACheckpointIfApplicable(rootSpace, 0);
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		MetaComputeStage *metaStage = stageSeqList->Nth(i);
		metaStage->constructComputationFlow(currentStageList, containerStage);
	}
	containerStage->addSyncStagesOnReturn(currentStageList);
	computation = containerStage;
	computation->setName("\"Computation Flow Specification\"");
	computation->reorganizeDynamicStages();
}

List<const char*> *ComputeSection::getRepeatIndexes() {
	List<const char*> *repeatIndexes = new List<const char*>;
	for (int i = 0; i < stageSeqList->NumElements(); i++) {
		MetaComputeStage *metaStage = stageSeqList->Nth(i);
		metaStage->populateRepeatIndexes(repeatIndexes);
	}
	List<const char*> *filteredList = new List<const char*>;
	for (int i = 0; i < repeatIndexes->NumElements(); i++) {
		const char* indexName = repeatIndexes->Nth(i);
		bool indexFound = false;
		for (int j = 0; j < filteredList->NumElements(); j++) {
			if (strcmp(filteredList->Nth(j), indexName) == 0) {
				indexFound = true;
				break;
			}
		}
		if (!indexFound) filteredList->Append(indexName);
	}	
	return filteredList;
}

void ComputeSection::performDependencyAnalysis(PartitionHierarchy *hierarchy) {
                computation->performDependencyAnalysis(hierarchy);
}
        
void ComputeSection::print() { computation->print(0); }

CompositeStage *ComputeSection::getComputation() { return computation; }
