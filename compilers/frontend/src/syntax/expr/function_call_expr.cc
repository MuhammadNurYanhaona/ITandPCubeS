#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------- Function Call -----------------------------------------------------/

FunctionCall::FunctionCall(Identifier *b, List<Expr*> *a, yyltype loc) : Expr(loc) {
        Assert(b != NULL && a != NULL);
        base = b;
        base->SetParent(this);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *expr = arguments->Nth(i);
                expr->SetParent(this);
        }
}

void FunctionCall::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Name) ");
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

Node *FunctionCall::clone() {
	Identifier *newBase = (Identifier*) base->clone();
	List<Expr*> *newArgs = new List<Expr*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *expr = arguments->Nth(i);
		newArgs->Append((Expr*) expr->clone());
	}
	return new FunctionCall(newBase, newArgs, *GetLocation());
}

void FunctionCall::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	Expr::retrieveExprByType(exprList, typeId);
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->retrieveExprByType(exprList, typeId);
	}	
}

int FunctionCall::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;
	bool allArgsResolved = true;
	List<Type*> *argTypeList = new List<Type*>;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		resolvedExprs += arg->resolveExprTypesAndScopes(scope);
		Type *argType = arg->getType();
		if (argType == NULL || argType == Type::errorType) {
			allArgsResolved = false;
		} else {
			argTypeList->Append(argType);
		}
	}

	if (allArgsResolved) {
		const char *functionName = base->getName();
		FunctionDef *fnDef = FunctionDef::fnDefMap->Lookup(functionName);
		if (fnDef == NULL) {
			ReportError::UndefinedSymbol(base, false);
			this->type = Type::errorType;
		} else {
			// determine the specific function instance for the type polymorphic function for
			// the current parameter types
			Scope *programScope = scope->get_nearest_scope(ProgramScope);
			Type *returnType = fnDef->resolveFnInstanceForParameterTypes(
					programScope, argTypeList, base);
			this->type = returnType;
			resolvedExprs++;
		}
	}

	delete argTypeList;
	return resolvedExprs;
}

int FunctionCall::emitSemanticErrors(Scope *scope) {
	int errors = 0;
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		errors += arg->emitScopeAndTypeErrors(scope);
	}
	return errors;
}

void FunctionCall::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->retrieveTerminalFieldAccesses(fieldList);
	}	
}

void FunctionCall::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	for (int i = 0; i < arguments->NumElements(); i++) {
                Expr *arg = arguments->Nth(i);
		arg->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

