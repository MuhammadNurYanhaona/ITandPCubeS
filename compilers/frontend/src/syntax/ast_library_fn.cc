#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "ast_library_fn.h"
#include "../common/errors.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"

#include <iostream>

//-------------------------------------------------------- Static Constants -----------------------------------------------------/

const char *Root::Name = "root";
const char *Random::Name = "random";
const char *LoadArray::Name = "load_array";
const char *StoreArray::Name = "store_array";
const char *BindInput::Name = "bind_input";
const char *BindOutput::Name = "bind_output";

//-------------------------------------------------------- Library Function -----------------------------------------------------/

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName,
                List<Expr*> *arguments, yyltype loc) : Expr(loc) {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments) : Expr() {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

bool LibraryFunction::isLibraryFunction(Identifier *id) {
        const char* name = id->getName();
        return (strcmp(name, Root::Name) == 0 
		|| strcmp(name, Random::Name) == 0
                || strcmp(name, LoadArray::Name) == 0 
                || strcmp(name, StoreArray::Name) == 0
                || strcmp(name, BindInput::Name) == 0
                || strcmp(name, BindOutput::Name) == 0);
}

LibraryFunction *LibraryFunction::getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc) {

        const char* name = id->getName();
        LibraryFunction *function = NULL;

        // note that there should never be a default 'else' block here; then the system will fail to find user 
	// defined functions
        if (strcmp(name, Root::Name) == 0) {
                function = new Root(id, arguments, loc);
        } else if (strcmp(name, Random::Name) == 0) {
                function = new Random(id, arguments, loc);
        } else if (strcmp(name, LoadArray::Name) == 0) {
                function = new LoadArray(id, arguments, loc);
        } else if (strcmp(name, StoreArray::Name) == 0) {
                function = new StoreArray(id, arguments, loc);
        } else if (strcmp(name, BindInput::Name) == 0) {
                function = new BindInput(id, arguments, loc);
        } else if (strcmp(name, BindOutput::Name) == 0) {
                function = new BindOutput(id, arguments, loc);
        }

        return function;
}

void LibraryFunction::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

Node *LibraryFunction::clone() {
	Identifier *newId = (Identifier*) functionName->clone();
	List<Expr*> *newArgsList = new List<Expr*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		Expr *arg = arguments->Nth(i);
		Expr *newArg = (Expr*) arg->clone();
		newArgsList->Append(newArg);
	}
	return getFunctionExpr(newId, newArgsList, *GetLocation());
}

void LibraryFunction::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
	}
}

//-------------------------------------------------------------- Root -----------------------------------------------------------/

int Root::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);
        Expr *arg2 = arguments->Nth(1);
        resolvedExprs += arg1->resolveExprTypes(scope);
        resolvedExprs += arg2->resolveExprTypes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::intType);

	Type *arg1Type = arg1->getType();
	if (arg1Type != NULL && arg2->getType() != NULL) {
		this->type = arg1Type;
		resolvedExprs++;
	}
	return resolvedExprs;
}

//--------------------------------------------------------- Array Operation -----------------------------------------------------/

int ArrayOperation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);
        resolvedExprs += arg1->resolveExprTypes(scope);
        Type *arg1Type = arg1->getType();

	Expr *arg2 = arguments->Nth(1);
        resolvedExprs += arg2->resolveExprTypes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::stringType);
        Type *arg2Type = arg2->getType();

	if (arg1Type != NULL && arg2Type != NULL) {
		this->type = Type::voidType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

//--------------------------------------------------------- Bind Operation ------------------------------------------------------/

int BindOperation::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	Expr *arg1 = arguments->Nth(0);	
	resolvedExprs += arg1->resolveExprTypes(scope);

	// identify the task environment type from the first argument
        Type *arg1Type = arg1->getType();
	NamedType *envType = NULL;
	if (arg1Type != NULL) {
                NamedType *objectType = dynamic_cast<NamedType*>(arg1Type);
                if (objectType != NULL && !objectType->isEnvironmentType()) {
                        envType = objectType;
                }
        }

	Expr *arg2 = arguments->Nth(1);
	resolvedExprs += arg2->resolveExprTypes(scope);
	resolvedExprs += arg2->performTypeInference(scope, Type::stringType);
	Type *arg2Type = arg2->getType();

	Expr *arg3 = arguments->Nth(2);
	resolvedExprs += arg3->resolveExprTypes(scope);
	resolvedExprs += arg3->performTypeInference(scope, Type::stringType);
	Type *arg3Type = arg3->getType();

	if (arg1Type != NULL && arg2Type != NULL && arg3Type != NULL) {
		this->type = Type::voidType;
		resolvedExprs++;
	}

	return resolvedExprs;
}
