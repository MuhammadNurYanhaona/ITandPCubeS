/* This header file holds all elements we need to annotate array assignment expressions properly to determine
   how to copy right hand side array to the left hand side. There are two possible ways of doing this: either
   we copy all data from the right to the left or we assign the reference of the right to the left. Given that
   we maintain separate metadata objects holding partition information of arrays, A simple 'array1 = array2'
   assignment needs to be translated as a series of statements contributing to transfering both data and meta-
   data from one to the other. We need to analyze each assignment expression involving array assingments to
   produce all information that will be needed during code generation to generate those statements.     
*/

#ifndef _array_assignment
#define _array_assignment

#include "../syntax/ast_expr.h"
#include "../syntax/ast_type.h"
#include "../semantics/scope.h"
#include "../utils/list.h"

enum AssignmentMode {COPY, REFERENCE};

// Access type info used to determine how to treat a particular dimension of the array during code generation.
enum DimensionAccessType {WHOLE, SUBRANGE, INDEX};

class ArrayName {
  protected:
	bool partOfEnv;
	const char *envObjName;
	const char *name;
	Type *type;
  public:
	ArrayName();
	void setPartOfEnv(bool partOfEnv);
	void setEnvObjName(const char *envObjName);
	void setName(const char *name);
	void setType(Type *type);
};

// a function to determine if an assignment expression involves assigning of an array to another so that we
// can treat it separately from more general case
bool isArrayAssignment(AssignmentExpr *expr);

// once we know an expression intends an array access, this function returns the name of the array been accessed
ArrayName *getArrayName(Expr *expr);

// Depending on actual dimensionalities of arrays on both sides we decide how the assignment should be carried
// on. This function makes that decision.   
AssignmentMode determineAssignmentMode(AssignmentExpr *expr, Scope *scope);

class DimensionAccess {
  protected:
	DimensionAccessType accessType;
	int dimensionNo;
	Expr *accessExpr;
  public:
};

class DimensionAnnotation {
  protected:
	DimensionAccess *assigneeInfo;
	DimensionAccess *assignerInfo;
  public:
};

class AssignmentDirective {
  protected:
	AssignmentExpr *expr;
	AssignmentMode mode;
	ArrayName *assigneeArray; 
	ArrayName *assignerArray;
	List<DimensionAnnotation*> *annotations;
  public:
	AssignmentDirective(AssignmentExpr *expr);
};

class AssignmentDirectiveList {
  protected:
	List<AssignmentDirective*> *directives;
  public:
	AssignmentDirectiveList(AssignmentExpr *expr);
};

#endif
