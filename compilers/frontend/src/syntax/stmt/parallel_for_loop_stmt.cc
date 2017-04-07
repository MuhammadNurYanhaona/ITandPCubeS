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
#include "../../semantics/loop_index.h"
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

	// if needed, create a new scope for the loop and enter it
	Scope *loopScope = NULL;
	if (iteration == 0) {
		loopScope = executionScope->enter_scope(new Scope(StatementBlockScope));
	} else {
		loopScope = executionScope->enter_scope(this->scope);
	}

	// in addition, if applicable, create a new index to array association scope and enter it
	if (iteration == 0) {
		IndexScope::currentScope->deriveNewScope();
		for (int i = 0; i < rangeConditions->NumElements(); i++) {
			IndexRangeCondition *cond = rangeConditions->Nth(i);
			cond->putIndexesInIndexScope();
		}
	} else {
		IndexScope::currentScope->enterScope(this->indexScope);
	}

	int resolvedExprs = 0;
	for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *condition = rangeConditions->Nth(i);
		resolvedExprs += condition->resolveExprTypesAndScopes(loopScope, iteration);
	}
	
	// try to resolve the body after evaluating the iteration expression to maximize type discovery
	resolvedExprs += body->resolveExprTypesAndScopes(loopScope, iteration);

	// exit the index association scope
	this->indexScope = IndexScope::currentScope;
        IndexScope::currentScope->goBackToOldScope();

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
                errors += condition->validateIndexAssociations(loopScope, this->indexScope);
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

List<LogicalExpr*> *PLoopStmt::getIndexRestrictions() {
        List<LogicalExpr*> *restrictionList = new List<LogicalExpr*>;
        for (int i = 0; i < rangeConditions->NumElements(); i++) {
                IndexRangeCondition *cond = rangeConditions->Nth(i);
                LogicalExpr *restriction = cond->getRestrictions();
                if (restriction != NULL) {
                        List<LogicalExpr*> *containedExprList = restriction->getANDBreakDown();
                        for (int j = 0; j < containedExprList->NumElements(); j++)
                        restrictionList->Append(containedExprList->Nth(j));
                }
        }
        return restrictionList;
}

