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
#include "../../codegen-helper/extern_config.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ Statement Block ---------------------------------------------------------/

StmtBlock::StmtBlock(List<Stmt*> *s) : Stmt() {
        Assert(s != NULL);
        stmts = s;
        for (int i = 0; i < stmts->NumElements(); i++) {
                stmts->Nth(i)->SetParent(this);
        }
}

void StmtBlock::PrintChildren(int indentLevel) {
        stmts->PrintAll(indentLevel + 1);
}

Node *StmtBlock::clone() {
	List<Stmt*> *newStmtList = new List<Stmt*>;
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		Stmt *newStmt = (Stmt*) stmt->clone();
		newStmtList->Append(newStmt);
        }
	return new StmtBlock(newStmtList);
}

void StmtBlock::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		stmt->retrieveExprByType(exprList, typeId);
	}
}

int StmtBlock::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	int resolvedExprs = 0;
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		resolvedExprs += stmt->resolveExprTypesAndScopes(executionScope, iteration);
	}
	return resolvedExprs;
}

int StmtBlock::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                errors += stmt->emitScopeAndTypeErrors(scope);
	}
	return errors;
}

void StmtBlock::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
		stmt->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

Hashtable<VariableAccess*> *StmtBlock::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
        Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                mergeAccessedVariables(table, stmt->getAccessedGlobalVariables(globalReferences));
        }
        return table;
}

void StmtBlock::analyseEpochDependencies(Space *space) {
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->analyseEpochDependencies(space);
        }
}

void StmtBlock::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
        }
}

void StmtBlock::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->retrieveExternHeaderAndLibraries(includesAndLinksMap);
        }
}

