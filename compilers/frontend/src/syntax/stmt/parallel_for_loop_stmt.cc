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
#include "../../semantics/data_access.h"
#include "../../static-analysis/reduction_info.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ Parallel Loop ----------------------------------------------------------/

PLoopStmt::PLoopStmt(List<IndexRangeCondition*> *rc, Stmt *b, yyltype loc) : LoopStmt(b, loc) {
        Assert(rc != NULL);
        rangeConditions = rc;
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                rangeConditions->Nth(i)->SetParent(this);
        }
}

void PLoopStmt::PrintChildren(int indentLevel) {
        rangeConditions->PrintAll(indentLevel + 1);
        body->Print(indentLevel + 1);
}

Node *PLoopStmt::clone() {
	List<IndexRangeCondition*> *newCondList = new List<IndexRangeCondition*>;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		IndexRangeCondition *newCond = (IndexRangeCondition*) condition->clone();
		newCondList->Append(newCond);
        }
	Stmt *newBody = (Stmt*) body->clone();
	return new PLoopStmt(newCondList, newBody, *GetLocation());
}

void PLoopStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		condition->retrieveExprByType(exprList, typeId);
	}
	body->retrieveExprByType(exprList, typeId);
}

int PLoopStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {

	// create a new scope for the loop and enter it
	Scope *loopScope = NULL;
	if (iteration == 0) {
		loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	} else {
		loopScope = executionScope->enter_scope(this->scope);
	}

	int resolvedExprs = 0;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		resolvedExprs += condition->resolveExprTypesAndScopes(loopScope, iteration);
	}
	
	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(loopScope, iteration);

	// exit the scope
	loopScope->detach_from_parent();
        this->scope = loopScope;

	return resolvedExprs;
}

int PLoopStmt::emitScopeAndTypeErrors(Scope *executionScope) {
	int errors = 0;
	Scope *loopScope = executionScope->enter_scope(this->scope);
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
                errors += condition->emitScopeAndTypeErrors(loopScope);
        }
        errors += body->emitScopeAndTypeErrors(loopScope);
	loopScope->detach_from_parent();
        return errors;
}

void PLoopStmt::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		condition->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
	body->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}

Hashtable<VariableAccess*> *PLoopStmt::getAccessedGlobalVariables(
		TaskGlobalReferences *globalReferences) {
        Hashtable<VariableAccess*> *table = body->getAccessedGlobalVariables(globalReferences);
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *cond = rangeConditions->Nth(i);
                mergeAccessedVariables(table, cond->getAccessedGlobalVariables(globalReferences));
        }
        return table;
}

void PLoopStmt::analyseEpochDependencies(Space *space) {
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *cond = rangeConditions->Nth(i);
                cond->analyseEpochDependencies(space);
        }
	body->analyseEpochDependencies(space);
}

