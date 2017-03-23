#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ While Loop ------------------------------------------------------------/

WhileStmt::WhileStmt(Expr *c, Stmt *b, yyltype loc) : Stmt(loc) {
        Assert(c != NULL && b != NULL);
        condition = c;
        condition->SetParent(this);
        body = b;
        body->SetParent(this);
}

void WhileStmt::PrintChildren(int indentLevel) {
        condition->Print(indentLevel + 1, "(Condition) ");
        body->Print(indentLevel + 1);
}

Node *WhileStmt::clone() {
	Expr *newCond = (Expr*) condition->clone();
	Stmt *newBody = (Stmt*) body->clone();
	return new WhileStmt(newCond, newBody, *GetLocation());
}

void WhileStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	condition->retrieveExprByType(exprList, typeId);
	body->retrieveExprByType(exprList, typeId);
}

int WhileStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	
	int resolvedExprs = 0;
	resolvedExprs += body->resolveExprTypesAndScopes(executionScope, iteration);
	
	// We resolve the condition after we evaluate the body because the IT while loop is
	// similar to a 'do {...} while (condition);' loop in C. We want the while condition 
	// to pick up any local variable defined in the loop body.
	resolvedExprs += condition->resolveExprTypesAndScopes(executionScope, iteration);
	resolvedExprs += condition->performTypeInference(executionScope, Type::boolType);

	return resolvedExprs;
}

int WhileStmt::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	errors += condition->emitScopeAndTypeErrors(scope);
	Type *condType = condition->getType();
	if (condType != NULL && condType != Type::boolType) {
		ReportError::InvalidExprType(condition, Type::boolType, false);
		errors++;
	}
        errors += body->emitScopeAndTypeErrors(scope);
	return errors;
}

void WhileStmt::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	condition->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	body->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}
