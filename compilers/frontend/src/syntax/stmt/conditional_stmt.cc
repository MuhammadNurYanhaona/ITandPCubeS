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

//-------------------------------------------------------- Conditional Statement -------------------------------------------------------/

ConditionalStmt::ConditionalStmt(Expr *c, Stmt *s, yyltype loc) : Stmt(loc) {
        Assert(s != NULL);
        condition = c;
        if (condition != NULL) {
                condition->SetParent(this);
        }
        stmt = s;
        stmt->SetParent(this);
}

void ConditionalStmt::PrintChildren(int indentLevel) {
        if (condition != NULL) condition->Print(indentLevel, "(If) ");
        stmt->Print(indentLevel);
}

Node *ConditionalStmt::clone() {
	Expr *newCond = NULL;
	if (condition != NULL) { 
		newCond = (Expr*) condition->clone();
	}
	Stmt *newStmt = (Stmt*) stmt->clone();
	return new ConditionalStmt(newCond, newStmt, *GetLocation());
}

void ConditionalStmt::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (condition != NULL) condition->retrieveExprByType(exprList, typeId);
	stmt->retrieveExprByType(exprList, typeId);
}

int ConditionalStmt::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	int resolvedExprs = stmt->resolveExprTypesAndScopes(executionScope, iteration);
	if (condition != NULL) {
		resolvedExprs += condition->resolveExprTypesAndScopes(executionScope, iteration);
		int inferredTypes = condition->performTypeInference(executionScope, Type::boolType);
		resolvedExprs += inferredTypes;
	}
	return resolvedExprs;
}

int ConditionalStmt::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	if (condition != NULL) {
		Type *condType = condition->getType();
		if (condType != NULL && condType != Type::boolType) {
			ReportError::InvalidExprType(condition, Type::boolType, false);
			errors++;
		} else {
			errors += condition->emitScopeAndTypeErrors(scope);
		}
	}
	errors += stmt->emitScopeAndTypeErrors(scope);
	return errors;
}

void ConditionalStmt::performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	if (condition != NULL) {
		condition->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	}
	stmt->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}

Hashtable<VariableAccess*> *ConditionalStmt::getAccessedGlobalVariables(
		TaskGlobalReferences *globalReferences) {
        Hashtable<VariableAccess*> *table = stmt->getAccessedGlobalVariables(globalReferences);
        if (condition != NULL) {
		mergeAccessedVariables(table, 
			condition->getAccessedGlobalVariables(globalReferences));
	}
        return table;
}

void ConditionalStmt::analyseEpochDependencies(Space *space) {
        if (condition != NULL) {
                condition->setEpochVersions(space, 0);
        }
        stmt->analyseEpochDependencies(space);
}

void ConditionalStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy,
                Space *executingLps) {
        stmt->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
}

void ConditionalStmt::retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {
        stmt->retrieveExternHeaderAndLibraries(includesAndLinksMap);
}
