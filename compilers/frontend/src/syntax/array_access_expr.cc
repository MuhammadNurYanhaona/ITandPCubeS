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

//--------------------------------------------------- Array Access ----------------------------------------------------/

ArrayAccess::ArrayAccess(Expr *b, Expr *i, yyltype loc) : Expr(loc) {
        Assert(b != NULL && i != NULL);
        base = b;
        base->SetParent(this);
        index = i;
        index->SetParent(this);
}

void ArrayAccess::PrintChildren(int indentLevel) {
        base->Print(indentLevel + 1, "(Base) ");
        index->Print(indentLevel + 1, "(Index) ");
}

Node *ArrayAccess::clone() {
	Expr *newBase = (Expr*) base->clone();
	Expr *newIndex = (Expr*) index->clone();
	return new ArrayAccess(newBase, newIndex, *GetLocation());
}

void ArrayAccess::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (typeId == getExprTypeId()) {
		Expr::retrieveExprByType(exprList, typeId);
	} else {
		base->retrieveExprByType(exprList, typeId);
		index->retrieveExprByType(exprList, typeId);
	}
}

int ArrayAccess::getIndexPosition() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) return precedingAccess->getIndexPosition() + 1;
        return 0;
}

Expr *ArrayAccess::getEndpointOfArrayAccess() {
        ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
        if (precedingAccess != NULL) {
                return precedingAccess->getEndpointOfArrayAccess();
        } else return base;
}

int ArrayAccess::resolveExprTypes(Scope *scope) {

	int resolvedExprs = 0;
	resolvedExprs += base->resolveExprTypesAndScopes(scope);
        Type *baseType = base->getType();
	if (baseType == NULL) return resolvedExprs;

	ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
	if (arrayType == NULL) {
		this->type = Type::errorType;
		return resolvedExprs;
	}

	IndexRange *indexRange = dynamic_cast<IndexRange*>(index);
        if (indexRange != NULL) {
		this->type = arrayType;
		resolvedExprs += indexRange->resolveExprTypesAndScopes(scope);
	} else {
		this->type = arrayType->reduceADimension();
		resolvedExprs += index->resolveExprTypesAndScopes(scope);
		resolvedExprs += index->performTypeInference(scope, Type::intType);	
	}
	resolvedExprs++;
	return resolvedExprs;
}

int ArrayAccess::emitSemanticErrors(Scope *scope) {
	
	int errors = 0;
	errors += base->emitScopeAndTypeErrors(scope);
        Type *baseType = base->getType();
        if (baseType == NULL) {
		ReportError::InvalidArrayAccess(GetLocation(), NULL, false);
        } else {
                ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
                if (arrayType == NULL) {
			ReportError::InvalidArrayAccess(base->GetLocation(), baseType, false);
		}
	}
	errors += index->emitScopeAndTypeErrors(scope);
	return errors;
}

