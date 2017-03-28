#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/data_access.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------- Epoch Expression --------------------------------------------------/

EpochExpr::EpochExpr(Expr *r, int lag) : Expr(*r->GetLocation()) {
        Assert(r != NULL && lag >= 0);
        root = r;
        root->SetParent(root);
        this->lag = lag;
}

void EpochExpr::PrintChildren(int indentLevel) {
        root->Print(indentLevel + 1, "(RootExpr) ");
        PrintLabel(indentLevel + 1, "Lag ");
	printf("%d", lag);
}

Node *EpochExpr::clone() {
	Expr *newRoot = (Expr*) root->clone();
	return new EpochExpr(newRoot, lag);
}

void EpochExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		root->retrieveExprByType(exprList, typeId);
	}
}

int EpochExpr::resolveExprTypes(Scope *scope) {

	int resolvedExprs = root->resolveExprTypesAndScopes(scope);
	Type *rootType = root->getType();

	if (rootType != NULL && rootType != Type::errorType) {
		this->type = rootType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int EpochExpr::inferExprTypes(Scope *scope, Type *assignedType) {
	this->type = assignedType;
	int errors = 1;
	errors += root->performTypeInference(scope, assignedType);
	return errors;
}

int EpochExpr::emitSemanticErrors(Scope *scope) {
	return root->emitScopeAndTypeErrors(scope);
}

void EpochExpr::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	root->retrieveTerminalFieldAccesses(fieldList);
}

void EpochExpr::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {
	root->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}

Hashtable<VariableAccess*> *EpochExpr::getAccessedGlobalVariables(
		TaskGlobalReferences *globalRefs) {
        return root->getAccessedGlobalVariables(globalRefs);
}

void EpochExpr::setEpochVersions(Space *space, int oldEpochValue) {
        root->setEpochVersions(space, lag);
}

