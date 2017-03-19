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

//----------------------------------------------- Logical Expression -------------------------------------------------/

LogicalExpr::LogicalExpr(Expr *l, LogicalOperator o, Expr *r, yyltype loc) : Expr(loc) {
        Assert(r != NULL);
        left = l;
        if (left != NULL) {
                left->SetParent(this);
        }
        op = o;
        right = r;
        right->SetParent(this);
}

void LogicalExpr::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case AND: printf("&&"); break;
                case OR: printf("||"); break;
                case NOT: printf("!"); break;
                case EQ: printf("=="); break;
                case NE: printf("!="); break;
                case GT: printf(">"); break;
                case LT: printf("<"); break;
                case GTE: printf(">="); break;
                case LTE: printf("<="); break;
        }
        if (left != NULL) left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}

Node *LogicalExpr::clone() {
	Expr *newRight = (Expr*) right->clone();
	if (left == NULL) return new LogicalExpr(NULL, op, newRight, *GetLocation());
	Expr *newLeft = (Expr*) left->clone();
	return new LogicalExpr(newLeft, op, newRight, *GetLocation());
}

void LogicalExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		right->retrieveExprByType(exprList, typeId);
		if (left != NULL) left->retrieveExprByType(exprList, typeId);
	}
}

int LogicalExpr::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	if (left != NULL) {
		resolvedExprs += left->resolveExprTypesAndScopes(scope);
	}
	resolvedExprs += right->resolveExprTypesAndScopes(scope);

	bool arithmaticOp = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
	if (arithmaticOp) {
		resolvedExprs += right->performTypeInference(scope, left->getType());
		resolvedExprs += left->performTypeInference(scope, right->getType());
	} else {
		resolvedExprs += right->performTypeInference(scope, Type::boolType);
		resolvedExprs += left->performTypeInference(scope, Type::boolType);
	}
	
	this->type = Type::boolType;
	resolvedExprs++;

	return resolvedExprs;
}

int LogicalExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	
	// check the validity of the component expressions
	errors += right->emitScopeAndTypeErrors(scope);
	Type *rightType = right->getType();
	Type *leftType = NULL;
	if (left != NULL) {
		errors += left->emitScopeAndTypeErrors(scope);
        	leftType = left->getType();
	}

	// check the validity of combining/using the component expression(s) in the spcecific way as done by
	// the current expression
	bool arithmaticOperator = (op == EQ || op == NE || op == GT || op == LT || op == GTE || op == LTE);
        if (arithmaticOperator) {
                if (leftType != NULL && !(leftType == Type::intType
                                || leftType == Type::floatType
                                || leftType == Type::doubleType
                                || leftType == Type::charType
                                || leftType == Type::errorType)) {
                        ReportError::UnsupportedOperand(left, leftType, "logical expression", false);
			errors++;
                }
                if (rightType != NULL && !(rightType == Type::intType
                                || rightType == Type::floatType
                                || rightType == Type::doubleType
                                || rightType == Type::charType
                                || rightType == Type::errorType)) {
                        ReportError::UnsupportedOperand(right, rightType, "logical expression", false);
			errors++;
                }
                if (leftType != NULL && rightType != NULL) {
                        if (!leftType->isAssignableFrom(rightType)
                                        && !rightType->isAssignableFrom(leftType)) {
                                ReportError::TypeMixingError(this, leftType, rightType,
                                                "logical expression", false);
				errors++;
                        }
                }
        } else {
                if (rightType != NULL && !rightType->isAssignableFrom(Type::boolType)) {
                        ReportError::IncompatibleTypes(right->GetLocation(), rightType, Type::boolType, false);
			errors++;
                }
                if (leftType != NULL && !leftType->isAssignableFrom(Type::boolType)) {
                        ReportError::IncompatibleTypes(left->GetLocation(), leftType, Type::boolType, false);
			errors++;
                }
        }
	return errors;
}

