#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../common/errors.h"
#include "../common/constant.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

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
	resolvedExprs += right->resolveExprTypes(scope);
	Type *rightType = right->getType();
	resolvedExprs += left->resolveExprTypes(scope);
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

