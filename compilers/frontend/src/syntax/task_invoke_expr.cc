#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../common/errors.h"
#include "../common/constant.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//--------------------------------------------- Named Multi-Argument -------------------------------------------------/

NamedMultiArgument::NamedMultiArgument(char *argName, List<Expr*> *argList, yyltype loc) : Node(loc) {
	Assert(argName != NULL && argList != NULL && argList->NumElements() > 0);
	this->argName = argName;
	this->argList = argList;
	for (int i = 0; i < argList->NumElements(); i++) {
		this->argList->Nth(i)->SetParent(this);
	}
}

void NamedMultiArgument::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, argName);
	argList->PrintAll(indentLevel + 2);
}

Node *NamedMultiArgument::clone() {
	char *newName = strdup(argName);
	List<Expr*> *newArgList = new List<Expr*>;
	for (int i = 0; i < argList->NumElements(); i++) {
		Expr *arg = argList->Nth(i);
                newArgList->Append((Expr*) arg->clone());
        }
	return new NamedMultiArgument(newName, newArgList, *GetLocation());
}

void NamedMultiArgument::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < argList->NumElements(); i++) {
                Expr *arg = argList->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
        }
}

int NamedMultiArgument::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	for (int i = 0; i < argList->NumElements(); i++) {
                Expr *arg = argList->Nth(i);
		resolvedExprs += arg->resolveExprTypes(scope);
	}
	return resolvedExprs;
}

//----------------------------------------------- Task Invocation ----------------------------------------------------/

TaskInvocation::TaskInvocation(List<NamedMultiArgument*> *invocationArgs, yyltype loc) : Expr(loc) {
	Assert(invocationArgs != NULL);
	this->invocationArgs = invocationArgs;
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
		NamedMultiArgument *arg = invocationArgs->Nth(i);
		arg->SetParent(this);
	}
}

void TaskInvocation::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Arguments");
	invocationArgs->PrintAll(indentLevel + 2);
}

Node *TaskInvocation::clone() {
	List<NamedMultiArgument*> *newInvokeArgs = new List<NamedMultiArgument*>;
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
		NamedMultiArgument *arg = invocationArgs->Nth(i);
		newInvokeArgs->Append((NamedMultiArgument*) arg->clone());
	}
	return new TaskInvocation(newInvokeArgs, *GetLocation());
}

void TaskInvocation::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		for (int i = 0; i < invocationArgs->NumElements(); i++) {
			NamedMultiArgument *arg = invocationArgs->Nth(i);
			arg->retrieveExprByType(exprList, typeId);
		}
	}
}

int TaskInvocation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;

	// check for the existance of a task and an environment argument in execute command
	const char *taskName = getTaskName();
	FieldAccess *environment = getEnvArgument();
	if (taskName == NULL || environment == NULL) {
		ReportError::UnspecifiedTaskToExecute(GetLocation(), false);
		this->type = Type::errorType;
		return resolvedExprs;
	}

	// check for a valid matching task definition to execute
	Symbol *symbol = scope->lookup(taskName);
        TaskSymbol *task = NULL;
        if (symbol == NULL) {
                ReportError::UndefinedSymbol(GetLocation(), taskName, false);
		this->type = Type::errorType;
		return resolvedExprs;
        } else {
                task = dynamic_cast<TaskSymbol*>(symbol);
                if (task == NULL) {
                        ReportError::WrongSymbolType(GetLocation(), taskName, "Task", false);
			this->type = Type::errorType;
			return resolvedExprs;
                }
	}

	// set up the type of the environment argument if it is not known already
	TaskDef *taskDef = (TaskDef*) task->getNode();
	TupleDef *envTuple = taskDef->getEnvTuple();
	resolvedExprs += environment->resolveExprTypes(scope);
	resolvedExprs += environment->performTypeInference(scope, new NamedType(envTuple->getId()));

	// if there are any initialization arguments then resolve them
	List<Type*> *initArgTypes = taskDef->getInitArgTypes();
	List<Expr*> *initArgs = getInitArguments();
	bool fullyResolved = true;
	if (initArgTypes->NumElements() == initArgs->NumElements()) {
		for (int i = 0; i < initArgs->NumElements(); i++) {
			Expr *expr = initArgs->Nth(i);
			Type *type = initArgTypes->Nth(i);
			resolvedExprs += expr->resolveExprTypes(scope);
			resolvedExprs += expr->performTypeInference(scope, type);
			if (expr->getType() == NULL) {
				fullyResolved = false;
			}
		}
	}

	// resolve the partition arguments as integers if exists
	List<Expr*> *partitionArgs = getPartitionArguments();
	for (int i =0; i < partitionArgs->NumElements(); i++) {
		Expr *arg = partitionArgs->Nth(i);
		resolvedExprs += arg->resolveExprTypes(scope);
		resolvedExprs += arg->performTypeInference(scope, Type::intType);
	}

	if (fullyResolved) {
		this->type = Type::voidType;
		resolvedExprs++;
	}
	return resolvedExprs;	
}

int TaskInvocation::emitSemanticErrors(Scope *scope) {

	if (type == NULL || type == Type::errorType) return 0;

	int errors = 0;
	const char *taskName = getTaskName();
	Symbol *task = (TaskSymbol*) scope->lookup(taskName);
	TaskDef *taskDef = (TaskDef*) task->getNode();
	
	// the environment argument should be of proper type specific to the task being invoked
	FieldAccess *environment = getEnvArgument();
	TupleDef *envTuple = taskDef->getEnvTuple();
	Type *expectedEnvType = new NamedType(envTuple->getId());
	Type *envType = environment->getType();
	if (!envType->isEqual(expectedEnvType)) {
		ReportError::IncompatibleTypes(environment->GetLocation(), envType,
					expectedEnvType, false);
		errors++;
	}

	// if there are any initialization arguments then the number should match the expected
	// arguments and the types should be appropriate
	List<Type*> *initArgTypes = taskDef->getInitArgTypes();
	List<Expr*> *initArgs = getInitArguments();
	if (initArgTypes->NumElements() != initArgs->NumElements()) {
		NamedMultiArgument *init = retrieveArgByName("initialize");
		Identifier *sectionId = new Identifier(*init->GetLocation(), "Initialize");
		ReportError::TooFewOrTooManyParameters(sectionId, initArgs->NumElements(),
				initArgTypes->NumElements(), false);
	} else {
		for (int i = 0; i < initArgTypes->NumElements(); i++) {
			Type *expected = initArgTypes->Nth(i);
			Expr *arg = initArgs->Nth(i);
			errors += arg->emitScopeAndTypeErrors(scope);
			Type *found = arg->getType();
			if (found != NULL && found != Type::errorType 
					&& !expected->isAssignableFrom(found)) {
				ReportError::IncompatibleTypes(arg->GetLocation(),
						found, expected, false);
				errors++;
			}
		}
	}
	
	// all partition arguments should be integers and their count should match expectation
	List<Expr*> *partitionArgs = getPartitionArguments();
	int argsCount = taskDef->getPartitionArgsCount();
	if (argsCount != partitionArgs->NumElements()) {
		NamedMultiArgument *partition = retrieveArgByName("partition");
		Identifier *sectionId = new Identifier(*partition->GetLocation(), "Partition");
		ReportError::TooFewOrTooManyParameters(sectionId, partitionArgs->NumElements(),
				argsCount, false);
	} else {
		for (int i = 0; i < partitionArgs->NumElements(); i++) {
			Expr *arg = partitionArgs->Nth(i);
			errors += arg->emitScopeAndTypeErrors(scope);
			Type *argType = arg->getType();
			if (argType != NULL && !Type::intType->isAssignableFrom(argType)) {
                                ReportError::IncompatibleTypes(arg->GetLocation(),
					argType, Type::intType, false);
                        }
		}
	}

	return errors;
}

const char *TaskInvocation::getTaskName() {
	NamedMultiArgument *nameArg = retrieveArgByName("task");
	if (nameArg == NULL) return NULL;
	List<Expr*> *argList = nameArg->getArgList();
	StringConstant *taskName = dynamic_cast<StringConstant*>(argList->Nth(0));
	if (taskName == NULL) return NULL;
	return taskName->getValue();
}

FieldAccess *TaskInvocation::getEnvArgument() {
	NamedMultiArgument *envArg = retrieveArgByName("environment");
	if (envArg == NULL) return NULL;
        List<Expr*> *argList = envArg->getArgList();
	Expr *firstArg = argList->Nth(0);
	return dynamic_cast<FieldAccess*>(firstArg);
}

List<Expr*> *TaskInvocation::getInitArguments() {
	NamedMultiArgument *initArg = retrieveArgByName("initialize");
	if (initArg == NULL) return new List<Expr*>;
        return initArg->getArgList();
}

List<Expr*> *TaskInvocation::getPartitionArguments() {
	NamedMultiArgument *partitionArg = retrieveArgByName("partition");
        if (partitionArg == NULL) return new List<Expr*>;
        return partitionArg->getArgList();
}

NamedMultiArgument *TaskInvocation::retrieveArgByName(const char *argName) {
	for (int i = 0; i < invocationArgs->NumElements(); i++) {
        	NamedMultiArgument *arg = invocationArgs->Nth(i);
		const char *currentArgName = arg->getName();
		if (strcmp(argName, currentArgName) == 0) return arg;
	}
	return NULL;
}
