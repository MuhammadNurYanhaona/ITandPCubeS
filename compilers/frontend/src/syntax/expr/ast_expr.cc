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

//--------------------------------------------------- Expression ------------------------------------------------------/

void Expr::retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {
	if (this->getExprTypeId() == typeId) {
		exprList->Append(this);
	}
}

int Expr::resolveExprTypesAndScopes(Scope *executionScope, int iteration) {
	if (this->type == NULL) {
		return resolveExprTypes(executionScope);
	}
	return 0;
}

int Expr::performTypeInference(Scope *executionScope, Type *assumedType) {	
	
	if (assumedType == NULL || assumedType == Type::errorType) return 0;

	if (this->type == NULL || this->type == Type::errorType) {
		return inferExprTypes(executionScope, assumedType);
	}
	return 0;
}

int Expr::emitScopeAndTypeErrors(Scope *scope) {
	int errors = 0;
	if (type == NULL || type == Type::errorType) {
		ReportError::UnknownExpressionType(this, false);
		errors++;
	}
	return errors + emitSemanticErrors(scope);	
}

void Expr::retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList) {}

