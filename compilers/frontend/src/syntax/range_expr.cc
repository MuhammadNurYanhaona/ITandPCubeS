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

//----------------------------------------------- Range Expressions --------------------------------------------------/

RangeExpr::RangeExpr(Identifier *i, Expr *r, Expr *s, yyltype loc) : Expr(loc) {
        Assert(i != NULL && r != NULL);
        index = new FieldAccess(NULL, i, *i->GetLocation());
        index->SetParent(this);
        range = r;
        range->SetParent(this);
        step = s;
        if (step != NULL) {
                step->SetParent(this);
        }
        loopingRange = true;
}

RangeExpr::RangeExpr(Expr *i, Expr *r, yyltype loc) : Expr(loc) {
        Assert(i != NULL && r != NULL);
	Assert(dynamic_cast<FieldAccess*>(i) != NULL);
        index = (FieldAccess*) i;
        index->SetParent(this);
        range = r;
        range->SetParent(this);
        step = NULL;
        loopingRange = false;
}

void RangeExpr::PrintChildren(int indentLevel) {
        index->Print(indentLevel + 1, "(Index) ");
        range->Print(indentLevel + 1, "(Range) ");
        if (step != NULL) step->Print(indentLevel + 1, "(Step) ");
}

Node *RangeExpr::clone() {
	Identifier *newId = (Identifier*) index->getField()->clone();
	Expr *newRange = (Expr*) range->clone();
	if (loopingRange) {
		Expr *newStep = NULL;
		if (step != NULL) {
			newStep = (Expr*) step->clone();
		}
		return new RangeExpr(newId, newRange, newStep, *GetLocation());
	}
	FieldAccess *newIndex = (FieldAccess*) index->clone(); 
	return new RangeExpr(newIndex, newRange, *GetLocation());
}

void RangeExpr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		index->retrieveExprByType(exprList, typeId);
		range->retrieveExprByType(exprList, typeId);
		if (step != NULL) step->retrieveExprByType(exprList, typeId);
	}	
}

int RangeExpr::resolveExprTypes(Scope *scope) {
	
	int resolvedExprs = 0;	

	resolvedExprs += index->resolveExprTypesAndScopes(scope);
	resolvedExprs += index->performTypeInference(scope, Type::intType);
	
	resolvedExprs += range->resolveExprTypesAndScopes(scope);
	resolvedExprs += range->performTypeInference(scope, Type::rangeType);

	if (step != NULL) {
		resolvedExprs += step->resolveExprTypesAndScopes(scope);
		resolvedExprs += step->performTypeInference(scope, Type::intType);
	}

	this->type = Type::boolType;
	resolvedExprs++;

	return resolvedExprs; 	
}

int RangeExpr::emitSemanticErrors(Scope *scope) {

	int errors = 0;
	errors += index->emitScopeAndTypeErrors(scope);
        Type *indexType = index->getType();
        if (indexType != NULL && indexType != Type::intType && indexType != Type::errorType) {
        	ReportError::IncompatibleTypes(index->GetLocation(), 
				indexType, Type::intType, false);
		errors++;
        }

	errors += range->emitScopeAndTypeErrors(scope);
	Type *rangeType = range->getType();
        if (rangeType != NULL && rangeType != Type::rangeType) {
                 ReportError::IncompatibleTypes(range->GetLocation(), 
				rangeType, Type::rangeType, false);
        }

	if (step != NULL) {
                errors += step->emitScopeAndTypeErrors(scope);
                Type *stepType = step->getType();
                if (stepType != NULL 
				&& !Type::intType->isAssignableFrom(stepType)) {
                        ReportError::IncompatibleTypes(step->GetLocation(), 
					stepType, Type::intType, false);
                }
        }

	return errors;
}

