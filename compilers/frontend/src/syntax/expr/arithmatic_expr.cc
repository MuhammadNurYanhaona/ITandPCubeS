#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_type.h"
#include "../../common/errors.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//---------------------------------------------- Arithmatic Expression ------------------------------------------------/

ArithmaticExpr::ArithmaticExpr(Expr *l, ArithmaticOperator o, Expr *r, yyltype loc) : Expr(loc) {
        Assert(l != NULL && r != NULL);
        left = l;
        left->SetParent(this);
        op = o;
        right = r;
        right->SetParent(this);
}

void ArithmaticExpr::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Operator");
        switch (op) {
                case ADD: printf("+"); break;
                case SUBTRACT: printf("-"); break;
                case MULTIPLY: printf("*"); break;
                case DIVIDE: printf("/"); break;
                case MODULUS: printf("%c", '%'); break;
                case LEFT_SHIFT: printf("<<"); break;
                case RIGHT_SHIFT: printf(">>"); break;
                case POWER: printf("**"); break;
                case BITWISE_AND: printf("&"); break;
                case BITWISE_XOR: printf("^"); break;
                case BITWISE_OR: printf("|"); break;
        }
        left->Print(indentLevel + 1);
        right->Print(indentLevel + 1);
}

Node *ArithmaticExpr::clone() {
	Expr *newLeft = (Expr*) left->clone();
	Expr *newRight = (Expr*) right->clone();
	return new ArithmaticExpr(newLeft, op, newRight, *GetLocation());
}

void ArithmaticExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		left->retrieveExprByType(exprList, typeId);
		right->retrieveExprByType(exprList, typeId);
	}
}

int ArithmaticExpr::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;
	resolvedExprs += left->resolveExprTypesAndScopes(scope);
        Type *leftType = left->getType();
        resolvedExprs += right->resolveExprTypesAndScopes(scope);
        Type *rightType = right->getType();

	if (leftType != NULL && rightType != NULL) {
                if (leftType->isAssignableFrom(rightType)) {
                        this->type = leftType;
			resolvedExprs++;
                } else if (rightType->isAssignableFrom(leftType)) {
                        this->type = rightType;
			resolvedExprs++;
		}
	}
	
	resolvedExprs += left->performTypeInference(scope, rightType);
	resolvedExprs += right->performTypeInference(scope, leftType);

	return resolvedExprs;
}

int ArithmaticExpr::inferExprTypes(Scope *scope, Type *assignedType) {
	this->type = assignedType;
	int resolvedExprs = 1;
	resolvedExprs += left->performTypeInference(scope, assignedType);
        resolvedExprs += right->performTypeInference(scope, assignedType);
	return resolvedExprs;
}

int ArithmaticExpr::emitSemanticErrors(Scope *scope) {
	
	int errors = 0;

	// check the validity of the left-hand-side expression and its use in the arithmatic expression
	errors += left->emitScopeAndTypeErrors(scope);
	Type *leftType = left->getType();
	if (leftType != NULL && !(leftType == Type::intType
                        || leftType == Type::floatType
                        || leftType == Type::doubleType
                        || leftType == Type::charType
                        || leftType == Type::errorType)) {
                ReportError::UnsupportedOperand(left, leftType, "arithmatic expression", false);
		errors++;
        }
	
	// check the validity of the right-hand-side expression and its use in the arithmatic expression
	errors += right->emitScopeAndTypeErrors(scope);
	Type *rightType = right->getType();
        if (rightType != NULL && !(rightType == Type::intType
                        || rightType == Type::floatType
                        || rightType == Type::doubleType
                        || rightType == Type::charType
                        || rightType == Type::errorType)) {
                ReportError::UnsupportedOperand(right, rightType, "arithmatic expression", false);
		errors++;
        }

	// check the validity of combining the left and right hand-side expressions in arithmatic
        if (op == BITWISE_AND || op == BITWISE_OR || op == BITWISE_XOR) {
                if ((leftType != NULL 
				&& !(leftType == Type::intType 
					|| leftType == Type::errorType))
                       		|| (rightType != NULL && !(rightType == Type::intType
                                        || rightType == Type::errorType))) {
                        ReportError::UnsupportedOperand(right,
                                        rightType, "arithmatic expression", false);
			errors++;
                }
        }
        if (leftType != NULL && rightType != NULL) {
                if (!leftType->isAssignableFrom(rightType) 
				&& !rightType->isAssignableFrom(leftType)) {
                        ReportError::TypeMixingError(this,
                                        leftType, rightType, "arithmatic expression", false);
			errors++;
                }
        }
	return errors;
}

void ArithmaticExpr::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {
	left->retrieveTerminalFieldAccesses(fieldList);
	right->retrieveTerminalFieldAccesses(fieldList);
}

void ArithmaticExpr::performStageParamReplacement(
		Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
		Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {

	left->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
	right->performStageParamReplacement(nameAdjustmentInstrMap, arrayAccXformInstrMap);
}
