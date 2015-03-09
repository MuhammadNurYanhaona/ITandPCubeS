#include "array_assignment.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"

bool isArrayAssignment(AssignmentExpr *expr) {
	Type *type = expr->getType();
	if (type == NULL) return false;
	ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
	StaticArrayType *staticType = dynamic_cast<StaticArrayType*>(type);
	return (arrayType != NULL && staticType == NULL);
}

ArrayName *getArrayName(Expr *expr) {
	FieldAccess *fieldAccess = NULL;
	Expr *currentExpr = expr;
	while ((fieldAccess = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
		ArrayAccess *array = dynamic_cast<ArrayAccess*>(currentExpr);
		currentExpr = array->getBase();	
	}
	ArrayName *arrayName = new ArrayName();

	// The only ways we can have an array is either as an stand-alone reference or as an element of
	// an environment object. This is because user defined types cannot have arrays as element. Thus
	// we can easily determine if an array is part of an environment or not by simply checking if 
	// corresponding field-access reference is a terminal field access. 
	bool partOfEnv = fieldAccess->isTerminalField();

	arrayName->setPartOfEnv(partOfEnv);
	arrayName->setName(fieldAccess->getField()->getName());
	if (partOfEnv) {
		FieldAccess *base = dynamic_cast<FieldAccess*>(fieldAccess->getBase());
		arrayName->setEnvObjName(base->getField()->getName());
	}
	arrayName->setType(fieldAccess->getType());

	return arrayName;
}
