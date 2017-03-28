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
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ If/Else Block -----------------------------------------------------------/

IfStmt::IfStmt(List<ConditionalStmt*> *ib, yyltype loc) : Stmt(loc) {
        Assert(ib != NULL);
        ifBlocks = ib;
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ifBlocks->Nth(i)->SetParent(this);
        }
}

void IfStmt::PrintChildren(int indentLevel) {
        ifBlocks->PrintAll(indentLevel + 1);
}

Node *IfStmt::clone() {
	List<ConditionalStmt*> *newBlocks = new List<ConditionalStmt*>;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		ConditionalStmt *newStmt = (ConditionalStmt*) stmt->clone();
		newBlocks->Append(newStmt);
        }
	return new IfStmt(newBlocks, *GetLocation());
}

void IfStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->retrieveExprByType(exprList, typeId);
	}
}

int IfStmt::resolveExprTypesAndScopes(Scope *excutionScope, int iteration) {
	int resolvedExprs = 0;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		resolvedExprs += stmt->resolveExprTypesAndScopes(excutionScope, iteration);
	}
	return resolvedExprs;
}

int IfStmt::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		errors += stmt->emitScopeAndTypeErrors(scope);
	}
	return errors;
}

void IfStmt::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	
	for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
		stmt->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
}

Hashtable<VariableAccess*> *IfStmt::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
        Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
                mergeAccessedVariables(table, stmt->getAccessedGlobalVariables(globalReferences));
        }
        return table;
}

void IfStmt::analyseEpochDependencies(Space *space) {
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
                stmt->analyseEpochDependencies(space);
        }
}

