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

//--------------------------------------------- Assignment Expression ------------------------------------------------/

AssignmentExpr::AssignmentExpr(Expr *l, Expr *r, yyltype loc) : Expr(loc) {
        Assert(l != NULL && r != NULL);
        left = l;
        left->SetParent(this);
        right = r;
        right->SetParent(this);
}

void AssignmentExpr::PrintChildren(int indentLevel) {
        left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}

Node *AssignmentExpr::clone() {
	Expr *newLeft = (Expr*) left->clone();
	Expr *newRight = (Expr*) right->clone();
	return new AssignmentExpr(newLeft, newRight, *GetLocation());
}

void AssignmentExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		left->retrieveExprByType(exprList, typeId);
		right->retrieveExprByType(exprList, typeId);
	}
}

int AssignmentExpr::resolveExprTypes(Scope *scope) {
	int resolvedExprs = 0;
	resolvedExprs += right->resolveExprTypesAndScopes(scope);
	Type *rightType = right->getType();
	resolvedExprs += left->resolveExprTypesAndScopes(scope);
	Type *leftType = left->getType();
	resolvedExprs += left->performTypeInference(scope, rightType);
	resolvedExprs += right->performTypeInference(scope, leftType);
	if (leftType != NULL && rightType != NULL && leftType->isAssignableFrom(rightType)) {
		this->type = leftType;
		resolvedExprs++;
	}
	return resolvedExprs;
}

int AssignmentExpr::inferExprTypes(Scope *scope, Type *assignedType) {
	this->type = assignedType;
	int resolvedExprs = 1;
	resolvedExprs += left->performTypeInference(scope, assignedType);
        resolvedExprs += right->performTypeInference(scope, assignedType);
	return resolvedExprs;
}

int AssignmentExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	errors += left->emitScopeAndTypeErrors(scope);
	errors += right->emitScopeAndTypeErrors(scope);

	// check if the two sides of the assignment are compatible with each other
	Type *leftType = left->getType();
	Type *rightType = right->getType();
	if (leftType != NULL && rightType != NULL && !leftType->isAssignableFrom(rightType)) {
                ReportError::TypeMixingError(this, leftType, rightType, "assignment", false);
		errors++;
        }

	// check if the left-hand side of the assignment is a valid receiver expression
	FieldAccess *fieldAccess = dynamic_cast<FieldAccess*>(left);
        ArrayAccess *arrayAccess = dynamic_cast<ArrayAccess*>(left);
        if (fieldAccess == NULL && arrayAccess == NULL) {
                EpochExpr *epochExpr = dynamic_cast<EpochExpr*>(left);
                if (epochExpr == NULL) {
                        ReportError::NonLValueInAssignment(left, false);
			errors++;
                } else {
                        Expr *epochRoot = epochExpr->getRootExpr();
                        fieldAccess = dynamic_cast<FieldAccess*>(epochRoot);
                        arrayAccess = dynamic_cast<ArrayAccess*>(epochRoot);
                        if (fieldAccess == NULL && arrayAccess == NULL) {
                                ReportError::NonLValueInAssignment(left, false);
				errors++;
                        }
                }
        }
	return errors;
}

void AssignmentExpr::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	left->retrieveTerminalFieldAccesses(fieldList);
	right->retrieveTerminalFieldAccesses(fieldList);
}

void AssignmentExpr::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	left->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	right->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}

Hashtable<VariableAccess*> *AssignmentExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
        const char* baseVarName = left->getBaseVarName();
        if (baseVarName == NULL) {
                ReportError::NotLValueinAssignment(GetLocation());
        }
        Hashtable<VariableAccess*> *table = left->getAccessedGlobalVariables(globalReferences);
        if (baseVarName != NULL && table->Lookup(baseVarName) != NULL) {
                VariableAccess *accessLog = table->Lookup(baseVarName);
                if(accessLog->isContentAccessed())
                        accessLog->getContentAccessFlags()->flagAsWritten();
                if (accessLog->isMetadataAccessed())
                        accessLog->getMetadataAccessFlags()->flagAsWritten();
        }

        Hashtable<VariableAccess*> *rTable = right->getAccessedGlobalVariables(globalReferences);
        Iterator<VariableAccess*> iter = rTable->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
		if (accessLog->isMetadataAccessed()) accessLog->getMetadataAccessFlags()->flagAsRead();
		if(accessLog->isContentAccessed()) accessLog->getContentAccessFlags()->flagAsRead();
                if (table->Lookup(accessLog->getName()) != NULL) {
                        table->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
                } else {
                        table->Enter(accessLog->getName(), accessLog, true);
                }
        }
        // check if any local/global reference is made to some global variable through the assignment expression and
        // take care of that
        FieldAccess *field = dynamic_cast<FieldAccess*>(left);
        if (field != NULL && field->isTerminalField()) {
                const char *rightSide = right->getBaseVarName();
                Type *rightType = right->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(rightType);
                if (rightSide != NULL && globalReferences->doesReferToGlobal(rightSide) && arrayType != NULL) {
                        if (!globalReferences->isGlobalVariable(baseVarName)) {
                                VariableSymbol *root = globalReferences->getGlobalRoot(rightSide);
                                globalReferences->setNewReference(baseVarName, root->getName());
                        } else {
                                accessLog = table->Lookup(baseVarName);
                                accessLog->markContentAccess();
                                accessLog->getContentAccessFlags()->flagAsRedirected();
                        }
                }
        }
        return table;
}

void AssignmentExpr::setEpochVersions(Space *space, int epoch) {
        left->setEpochVersions(space, epoch);
        right->setEpochVersions(space, epoch);
}

