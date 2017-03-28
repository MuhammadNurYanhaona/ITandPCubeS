#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ Return Stmt -----------------------------------------------------------/

ReturnStmt::ReturnStmt(Expr *expr, yyltype loc) : Stmt(loc) {
	Assert(expr != NULL);
	this->expr = expr;
	this->expr->SetParent(this);
}

void ReturnStmt::PrintChildren(int indentLevel) {
	expr->Print(indentLevel + 1);
}

Node *ReturnStmt::clone() {
	Expr *newExpr = (Expr*) expr->clone();
	return new ReturnStmt(newExpr, *GetLocation());
}

void ReturnStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	expr->retrieveExprByType(exprList, typeId);
}

int ReturnStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	
	int resolvedExprs = expr->resolveExprTypesAndScopes(executionScope, iteration);
	
	// if the return type is resolved then assign it to the function that embodies this statement
	Type *returnType = expr->getType();
	if (returnType != NULL && returnType != Type::errorType) {
		FunctionInstance *fnInstance = FunctionInstance::getMostRecentFunction();
		if (fnInstance == NULL) {
			ReportError::ReturnStmtOutsideFn(GetLocation(), false);
		} else {
			Type *oldReturnType = fnInstance->getReturnType();
			if (oldReturnType == NULL 
					|| returnType->isAssignableFrom(oldReturnType)) {
				fnInstance->setReturnType(returnType);
			} else if (!oldReturnType->isAssignableFrom(returnType)) {
				ReportError::ConflictingReturnTypes(GetLocation(), 
						oldReturnType, returnType, false);
			}
		}
	}

	return resolvedExprs;
}

int ReturnStmt::emitScopeAndTypeErrors(Scope *scope) {
	return expr->emitScopeAndTypeErrors(scope);
}

void ReturnStmt::analyseEpochDependencies(Space *space) {
	expr->setEpochVersions(space, 0);	
}
