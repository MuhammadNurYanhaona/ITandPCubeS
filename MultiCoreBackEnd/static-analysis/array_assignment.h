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
	ArrayType *type;
  public:
	ArrayName();
	void setPartOfEnv(bool partOfEnv) { this->partOfEnv = partOfEnv; }
	void setEnvObjName(const char *envObjName) { this->envObjName = envObjName; }
	void setName(const char *name) { this->name = name; }
	void setType(ArrayType *type) { this->type = type; }
	ArrayType *getType() { return type; }
	void describe(int indent);
};

// a function to determine if an assignment expression involves assigning of an array to another so that we can 
// treat it separately from more general case
bool isArrayAssignment(AssignmentExpr *expr);

// once we know an expression intends an array access, this function returns the name of the array been accessed
ArrayName *getArrayName(Expr *expr);

// Depending on actual dimensionalities of arrays on both sides we decide how the assignment should be carried 
// on. This function makes that decision.   
AssignmentMode determineAssignmentMode(AssignmentExpr *expr);

// as its name suggests this class stores information regarding how a specific dimension of an array participating
// in an assignment expression is been accessed
class DimensionAccess {
  protected:
	DimensionAccessType accessType;
	int dimensionNo;
	Expr *accessExpr;
  public:
	DimensionAccess(int dimensionNo);
	DimensionAccess(Expr *accessExpr, int dimensionNo);	
	DimensionAccessType getAccessType() { return accessType; }
	int getDimensionNo() { return dimensionNo; }
	Expr *getAccessExpr() { return accessExpr; }
	bool isSingleEntry() { return accessType == INDEX; }
	void describe(int indent);
};

// Given an expression accessing an array, this function generates information regarding how various dimensions
// of the array have been accessed within the expression.	
List<DimensionAccess*> *generateDimensionAccessInfo(ArrayName *array, Expr *expr);

// This class stores information regarding how to get to the two dimensions been accessed on the two opposite 
// side of an assignment statements where the right dimension should be assigned to the left. To clarify the
// concept, let us take the example of the statement, a[i][...][j] = b[...][k..l]. If a is 4D and b is a 2D array
// in this statement then the second dimension of a should get elements from the first dimension of b and the 
// fourth of a should get part of the second dimension of b. This mapping from 2-to-1 and 4-to-2 must be retrieved
// to correctly translate the assignment statement. This class retains the information for one such mapping.  
class DimensionAnnotation {
  protected:
	// We have a list of dimension access info for both sides instead of a single one as whole/partial 
	// dimension accesses on both sides can be preceeded by arbitrary number of access to particular entry
	// in other dimensions.
	List<DimensionAccess*> *assigneeInfo;
	List<DimensionAccess*> *assignerInfo;

	// it may happen that the last sequence of dimension accesses on either side of the assignment statement
	// refer to particular indexes of underlying arrays. We need to store their information too for proper
	// code generation. The following flag in this class, when active, refers to the case where the annotation
	// is for such terminal sequences.  
	bool placeHolder;
  public:
	DimensionAnnotation(bool placeHolder) {
		this->placeHolder = placeHolder;
		this->assigneeInfo = NULL;
		this->assignerInfo = NULL;
	}
	bool isPlaceHolder() { return placeHolder; }
	void setAssigneeInfo(List<DimensionAccess*> *assigneeInfo) { this->assigneeInfo = assigneeInfo; }
	List<DimensionAccess*> *getAssigneeInfo() { return assigneeInfo; }
	void setAssignerInfo(List<DimensionAccess*> *assignerInfo) { this->assignerInfo = assignerInfo; }
	List<DimensionAccess*> *getAssignerInfo() { return assignerInfo; }	
	void describe(int indent);
};

// This class contains all information regarding appropriately translating a single array assignment expression
class AssignmentDirective {
  protected:
	// a reference to the assignment expression
	AssignmentExpr *expr;
	// the mode, data-copy/reference-assingment, selected for the expression
	AssignmentMode mode;
	// identity information of arrays on both sides
	ArrayName *assigneeArray; 
	ArrayName *assignerArray;
	// instructions regarding how to generate code for the assignment
	List<DimensionAnnotation*> *annotations;
  public:
	AssignmentDirective(AssignmentExpr *expr);
	void generateAnnotations();
	void describe(int indent);
};

// An assignment statement may be a composite of the form 'a = b = c' so we need a list of directives to hold 
// information regarding possibly multiple array transfers due to a single statement. Thus we have this class.
class AssignmentDirectiveList {
  protected:
	// a reference to the, possibly, composite original assignment expression
	AssignmentExpr *mainExpr;
	// a list holding directives regarding array transfer for each component assingment expression
	List<AssignmentDirective*> *directives;
  public:
	AssignmentDirectiveList(AssignmentExpr *expr);
	void describe(int indent);
};

#endif
