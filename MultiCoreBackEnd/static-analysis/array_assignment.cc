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
	arrayName->setType((ArrayType*) fieldAccess->getType());

	return arrayName;
}

AssignmentMode determineAssignmentMode(AssignmentExpr *expr) {
	
	Expr *left = expr->getLeft();
	ArrayName *leftArray = getArrayName(left);
	Type *leftType = leftArray->getType();
	Expr *right = expr->getRight();
	ArrayName *rightArray = getArrayName(right);
	Type *rightType = rightArray->getType();
	Type *exprType = expr->getType();

	// We choose to do a reference assignment from left to right side for an assignment expression 
	// only if the arrays on both sides have the same dimensionality and they are accessed in full
	// as part of the assignment expression.  
	if (leftType->isEqual(rightType) && leftType->isEqual(exprType)) return REFERENCE;
	// Otherwise, we copy data from right to the left side array.
	return COPY;
}

List<DimensionAccess*> *generateDimensionAccessInfo(ArrayName *array, Expr *expr) {
	
	int arrayDimensions = array->getType()->getDimensions();
	List<DimensionAccess*> *accessList = new List<DimensionAccess*>;

	// First we add default entries in the access list assuming that each dimension of the array 
	// has been explicitly accessed in the expression. This is needed as just an standalone reference
	// such as in 'a = b[...][i]' should be treated as accessing the first dimension of a wholly. 
	for (int i = 0; i < arrayDimensions; i++) {
		accessList->Append(new DimensionAccess(i));
	}

	// then we track down the dimensions actually been accessed in the expression
	List<Expr*> *accessExprList = new List<Expr*>;
	FieldAccess *fieldAccess = NULL;
	Expr *currentExpr = expr;
	while ((fieldAccess = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
		ArrayAccess *arrayAccess = dynamic_cast<ArrayAccess*>(currentExpr);
		currentExpr = arrayAccess->getBase();
		// since the traversal moves backward, elements should be added at the front of the list
		accessExprList->InsertAt(arrayAccess->getIndex(), 0);	
	}

	// then we replace entries in the original list based on the expressions retrieved
	for (int i = 0; i < accessExprList->NumElements(); i++) {
		DimensionAccess *actualAccessInfo = new DimensionAccess(accessExprList->Nth(i), i);
		accessList->RemoveAt(i);
		accessList->InsertAt(actualAccessInfo, i);
	} 

	return accessList;
}

//-------------------------------------------------------- Dimension Access -----------------------------------------------------/

DimensionAccess::DimensionAccess(int dimensionNo) {
	this->accessType = WHOLE;
	this->accessExpr = NULL;
	this->dimensionNo = dimensionNo;
}

DimensionAccess::DimensionAccess(Expr *accessExpr, int dimensionNo) {
	this->accessExpr = accessExpr;
	SubRangeExpr *subRange = dynamic_cast<SubRangeExpr*>(accessExpr);
	if (subRange != NULL) accessType = INDEX;
	else if (subRange->isFullRange()) {
		accessType = SUBRANGE;
	} else accessType = WHOLE;
	this->dimensionNo = dimensionNo;
}

//------------------------------------------------------- Assignment Directive --------------------------------------------------/

AssignmentDirective::AssignmentDirective(AssignmentExpr *expr) {
	this->expr = expr;
	this->mode = determineAssignmentMode(expr);
	this->annotations = new List<DimensionAnnotation*>;
	Expr *left = expr->getLeft();
	this->assigneeArray = getArrayName(left);
	Expr *right = expr->getRight();
	this->assignerArray = getArrayName(right);
}

void AssignmentDirective::generateAnnotations() {

	// get information about how different dimensions are accessed within the arrays present on two sides
	List<DimensionAccess*> *leftSideAccesses 
			= generateDimensionAccessInfo(assigneeArray, expr->getLeft());
	List<DimensionAccess*> *rightSideAccesses 
			= generateDimensionAccessInfo(assignerArray, expr->getRight());

	// then traverse over a list and generate annotation for each dimension transfer from right to left
	int i = 0;
	int j = 0;
	List<DimensionAccess*> *leftAccessList = new List<DimensionAccess*>;
	for (; i < leftSideAccesses->NumElements(); i++) {
		
		DimensionAccess *currentAccess = leftSideAccesses->Nth(i);
		leftAccessList->Append(currentAccess);
		if (currentAccess->isSingleEntry()) continue;

		// once a whole/partial dimension access is found for the left side, determine how to get to
		// the corresponding dimension on the right side
		List<DimensionAccess*> *rightAccessList = new List<DimensionAccess*>;
		for (; j < rightSideAccesses->NumElements(); j++) {
			DimensionAccess *rightAccess = rightSideAccesses->Nth(j);
			rightAccessList->Append(rightAccess);
			if (!rightAccess->isSingleEntry()) break;
		}
		
		// then create a dimension annotation and store that for current mapping
		DimensionAnnotation *annotation = new DimensionAnnotation(false);
		annotation->setAssigneeInfo(leftAccessList);
		annotation->setAssignerInfo(rightAccessList);

		// then refresh the left side list for next annotation and add the current annotation in the list
		leftAccessList = new List<DimensionAccess*>;
		annotations->Append(annotation);
	}

	// Finally, if there are entries left in the left and/or right expression list then generate a place 
	// holder annotation for it and store it in the list too.
	for (; i < leftSideAccesses->NumElements(); i++) leftAccessList->Append(leftSideAccesses->Nth(i));
	List<DimensionAccess*> *rightAccessList = new List<DimensionAccess*>;
	for (; j < rightSideAccesses->NumElements(); j++) rightAccessList->Append(rightSideAccesses->Nth(j));
	if (leftAccessList->NumElements() > 0 || rightAccessList->NumElements() > 0) {
		DimensionAnnotation *annotation = new DimensionAnnotation(true);
		annotation->setAssigneeInfo(leftAccessList);
		annotation->setAssignerInfo(rightAccessList);
		annotations->Append(annotation);
	}
}

//---------------------------------------------------- Assignment Directive List ------------------------------------------------/

AssignmentDirectiveList::AssignmentDirectiveList(AssignmentExpr *expr) {
	
	this->mainExpr = expr;
	
	AssignmentExpr *rightAssignment = NULL;
	Expr *currentRight = expr;
	List<Expr*> *leftList = new List<Expr*>;	
	while ((rightAssignment = dynamic_cast<AssignmentExpr*>(currentRight)) != NULL) {
		leftList->Append(rightAssignment->getLeft());
		currentRight = rightAssignment->getRight();
	}
	
	List<AssignmentExpr*> *assignmentList = new List<AssignmentExpr*>;
	for (int i = 0; i < leftList->NumElements(); i++) {
		Expr *leftExpr = leftList->Nth(i);
		AssignmentExpr *assignment = new AssignmentExpr(leftExpr, currentRight, *leftExpr->GetLocation());
		assignment->setType(expr->getType());
		assignmentList->Append(assignment);
	}

	this->directives = new List<AssignmentDirective*>;
	for (int i = 0; i < assignmentList->NumElements(); i++) {
		AssignmentExpr *assignment = assignmentList->Nth(i);
		AssignmentDirective *directive = new AssignmentDirective(assignment);
		directive->generateAnnotations();
		this->directives->Append(directive);
	}
}
