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

//------------------------------------------------ Named Argument ----------------------------------------------------/

NamedArgument::NamedArgument(char *argName, Expr *argValue, yyltype loc) : Node(loc) {
	Assert(argName != NULL && argValue != NULL);
	this->argName = argName;
	this->argValue = argValue;
	this->argValue->SetParent(this);
}

void NamedArgument::PrintChildren(int indentLevel) {
	argValue->Print(indentLevel, argName);
}

Node *NamedArgument::clone() {
	char *newName = strdup(argName);
	Expr *newValue = (Expr*) argValue->clone();
	return new NamedArgument(newName, newValue, *GetLocation());
}

void NamedArgument::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	argValue->retrieveExprByType(exprList, typeId);
}

//------------------------------------------------ Object Create -----------------------------------------------------/

ObjectCreate::ObjectCreate(Type *o, List<NamedArgument*> *i, yyltype loc) : Expr(loc) {
        Assert(o != NULL && i != NULL);
        objectType = o;
        objectType->SetParent(this);
        initArgs = i;
        for (int j = 0; j < initArgs->NumElements(); j++) {
                initArgs->Nth(j)->SetParent(this);
        }
}

void ObjectCreate::PrintChildren(int indentLevel) {
        objectType->Print(indentLevel + 1);
        PrintLabel(indentLevel + 1, "Init-Arguments");
        initArgs->PrintAll(indentLevel + 2);
}

Node *ObjectCreate::clone() {
	Type *newType = (Type*) objectType->clone();
	List<NamedArgument*> *newArgsList = new List<NamedArgument*>;
	for (int j = 0; j < initArgs->NumElements(); j++) {
                NamedArgument *arg = initArgs->Nth(j);
		newArgsList->Append((NamedArgument*) arg->clone());
        }
	return new ObjectCreate(newType, newArgsList, *GetLocation());
}

void ObjectCreate::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int j = 0; j < initArgs->NumElements(); j++) {
                NamedArgument *arg = initArgs->Nth(j);
		arg->retrieveExprByType(exprList, typeId);
	}
}

int ObjectCreate::resolveExprTypes(Scope *scope) {

	// check if the object corresponds to an array or a list type; then component elements must be of a known type
	ArrayType *arrayType = dynamic_cast<ArrayType*>(objectType);
	ListType *listType = dynamic_cast<ListType*>(objectType);
	Type *terminalType = NULL;
	if (arrayType != NULL) terminalType = arrayType->getTerminalElementType();
	else if (listType != NULL) terminalType = listType->getTerminalElementType();
	if (terminalType != NULL) {
		if (scope->lookup(terminalType->getName()) == NULL) {
			this->type = Type::errorType;
			return 0;
		} else {
			this->type = objectType;
			return 1;
		}
	}
	
	// check if the object correspond to some built-in or user defined type; then the arguments will be properties
	const char* typeName = objectType->getName();
	Symbol *symbol = scope->lookup(typeName);
	if (symbol != NULL) {
		int resolvedExprs = 0;
        	TupleSymbol *tuple = dynamic_cast<TupleSymbol*>(symbol);
		if (tuple == NULL) {
			this->type = Type::errorType;
                        return 0;
		}
		List<Type*> *elementTypes = tuple->getElementTypes();
		if (initArgs->NumElements() > elementTypes->NumElements()) {
			ReportError::TooManyParametersInNew(GetLocation(), typeName, 
					initArgs->NumElements(),
					elementTypes->NumElements(), false);
		} 
		TupleDef *tupleDef = tuple->getTupleDef();
		const char *typeName = tupleDef->getId()->getName();
		bool fullyResolved = true;
		for (int i = 0; i < initArgs->NumElements(); i++) {
			NamedArgument *currentArg = initArgs->Nth(i);
			const char *argName = currentArg->getName();
			Expr *argValue = currentArg->getValue();
			resolvedExprs += argValue->resolveExprTypesAndScopes(scope);
			VariableDef *propertyDef = tupleDef->getComponent(argName);
        		if (propertyDef == NULL) {
				ReportError::InvalidInitArg(GetLocation(), typeName, argName, false);	
			} else {
				resolvedExprs += argValue->performTypeInference(scope, propertyDef->getType());
			}
			if (argValue->getType() == NULL) {
				fullyResolved = false;
			}
		}
		// We flag the current object creation expression to have a type only if all arguments are resolved
		// to allow type resolution process to make progress and the arguments to be resolved in some later
		// iteration of the scope-and-type checking.
		if (fullyResolved) {
			this->type = objectType;
			resolvedExprs++;
		}
		return resolvedExprs;
	}

	// the final option is that the object creation is correspond to the creation of an environment object for a
	// task
	if (strcmp(typeName, "TaskEnvironment") == 0) {
		if (initArgs->NumElements() != 1) {
                        ReportError::TaskNameRequiredInEnvironmentCreate(objectType->GetLocation(), false);
                        this->type = Type::errorType;
			return 0;
                }
		NamedArgument *arg = initArgs->Nth(0);
		StringConstant *str = dynamic_cast<StringConstant*>(arg->getValue());
		const char *argName = arg->getName();
		if (str == NULL || strcmp(argName, "name") != 0) {
                        ReportError::TaskNameRequiredInEnvironmentCreate(objectType->GetLocation(), false);
			this->type = Type::errorType;
			return 0;
		}
		Symbol *symbol = scope->lookup(str->getValue());
		if (symbol == NULL) {
			ReportError::UndefinedTask(str->GetLocation(), str->getValue(), false);
			this->type = Type::errorType;
			return 0;
		} else {
			TaskSymbol *task = dynamic_cast<TaskSymbol*>(symbol);
			if (task == NULL) {
				ReportError::UndefinedTask(str->GetLocation(), str->getValue(), false);
				this->type = Type::errorType;
				return 0;
			} else {
				TaskDef *taskDef = (TaskDef*) task->getNode();
				NamedType *envType = new NamedType(taskDef->getEnvTuple()->getId());
				envType->flagAsEnvironmentType();
				envType->setTaskName(str->getValue());
				this->type = envType;
				return 1;
			}
		}
	}
	
	// if none of the previous conditions matches then this object creation is in error and flaged as such
	this->type = Type::errorType;
	return 0;
}


