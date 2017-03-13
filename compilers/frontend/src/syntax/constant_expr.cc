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

//----------------------------------------------- Constant Expression -------------------------------------------------/

IntConstant::IntConstant(yyltype loc, int val) : Expr(loc) {
        value = val;
	size = TWO_BYTES;
	type = Type::intType;
}

IntConstant::IntConstant(yyltype loc, int value, IntSize size) : Expr(loc) {
        this->value = value;
        this->size = size;
	type = Type::intType;
}

void IntConstant::PrintChildren(int indentLevel) {
        printf("%d", value);
}

FloatConstant::FloatConstant(yyltype loc, float val) : Expr(loc) {
        value = val;
	type = Type::floatType;
}

void FloatConstant::PrintChildren(int indentLevel) {
        printf("%f", value);
}

DoubleConstant::DoubleConstant(yyltype loc, double val) : Expr(loc) {
        value = val;
	type = Type::doubleType;
}

void DoubleConstant::PrintChildren(int indentLevel) {
        printf("%g", value);
}

BoolConstant::BoolConstant(yyltype loc, bool val) : Expr(loc) {
        value = val;
	type = Type::boolType;
}

void BoolConstant::PrintChildren(int indentLevel) {
        printf("%s", value ? "true" : "false");
}

StringConstant::StringConstant(yyltype loc, const char *val) : Expr(loc) {
        Assert(val != NULL);
        value = strdup(val);
	type = Type::stringType;
}

void StringConstant::PrintChildren(int indentLevel) {
        printf("%s",value);
}

CharConstant::CharConstant(yyltype loc, char val) : Expr(loc) {
        value = val;
	type = Type::charType;
}

void CharConstant::PrintChildren(int indentLevel) {
        printf("%c",value);
}

ReductionVar::ReductionVar(char spaceId, const char *name, yyltype loc) : Expr(loc) {
	Assert(name != NULL);
	this->spaceId = spaceId;
	this->name = name;
}

void ReductionVar::PrintChildren(int indentLevel) {
	printf("Space %c:%s", spaceId, name);
}

int ReductionVar::resolveExprTypes(Scope *scope) {
	VariableSymbol *symbol = (VariableSymbol*) scope->lookup(name);
	if (symbol != NULL) {
		this->type = symbol->getType();
		return 1;
	}
	return 0;
}

int ReductionVar::emitSemanticErrors(Scope *scope) {

	Symbol *symbol = scope->lookup(name);
        if (symbol == NULL) {
		ReportError::UndefinedSymbol(GetLocation(), name, false);
		return 1;
	}
	VariableSymbol *varSymbol = dynamic_cast<VariableSymbol*>(symbol);
	if (varSymbol == NULL) {
		ReportError::WrongSymbolType(GetLocation(), name, "Reduction Variable", false);
		return 1;
	} else if (!varSymbol->isReduction()) {
		Identifier *varId = new Identifier(*GetLocation(), name);
		ReportError::NotReductionType(varId, false);
		return 1;
	}
	return 0;
}

