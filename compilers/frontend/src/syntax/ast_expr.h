#ifndef _H_ast_expr
#define _H_ast_expr

#include "ast.h"
#include "ast_stmt.h"
#include "ast_type.h"

#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>
#include <string.h>

class TaskDef;
class FieldAccess;
class Space;
class Scope;
class ParamReplacementConfig;
class VariableAccess;
class TaskGlobalReferences;

class Expr : public Stmt {
  protected:
	Type *type;
  public:
    	Expr(yyltype loc) : Stmt(loc) { type = NULL; }
    	Expr() : Stmt() { type = NULL; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	Type *getType() { return type; }

	// subclasses should implement this function to return a unique expression ID per class 
	virtual ExprTypeId getExprTypeId() = 0;

	// Any expresssion that may have some different kind of expression as its component sub-expression
	// must provides its own implementation for this function. Otherwise this baseclass implementation
	// is sufficient.
	virtual void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);

	// the interface function for scope-and-type checking inherited from the stmt base class; subclasses
	// should provide implementation for the protected function resolveExprTypes(Scope *scope) that does
	// all the work
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration = 0);

	// Sometimes the type of an expression can be assumed from the context it has been used. Similarly,
	// a larger parent expression my lead to type discoveries of its sub-expressions. This function is
	// the interface for this kind of type inference. The return value should indicate the number of 
	// new type resolutions have been enabled through the type inference mechanism. 
	int performTypeInference(Scope *executionScope, Type *assumedType);
	
	// The interface function for the same function from the stmt superclass; this takes care of the
	// case where the current expression is in error or type-less. Sub-classes should provide 
	// implementation for the emitSemanticErrors() function for more elaboration of error cases.
	int emitScopeAndTypeErrors(Scope *scope);

	// Terminal field accesses are access to base variables -- not to propertiers. This method is needed
	// for scope and other usage validation of variable accesses within a code.
	virtual void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);

	// An interface inherited from the statement superclass for resolving type polymorphic compute stages. 
	// Subclasses should provide a meaningful recursive implementation for this function so that all 
	// field and array accesses done on the underlying compute stage parameters are updated properly.
	virtual void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {}

	// This function decides, as its name suggests, the global variables been accessed by the expression.
        // It can track global arrays' reference assignments to some local variables and then indirect changes 
	// to the global arrays through these local references. This analysis is required to determine if
	// accessing a global variable is allowed from a compute stage to be executed on a particular LPS.
        virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);

	// This function finds out the root object within which an element been accessed or modified by some
        // expression. It makes sense only for reduction var, array-access, and field-access type expression. 
	// The function is however added to the common expression class to simplify any recursive base 
	// variable identification process in those expression subclasses.
        virtual const char *getBaseVarName() { return NULL; }
  protected:
	// The type resolution function that subclasses should implement; this gets called only when the
	// current expression type is NULL
	virtual int resolveExprTypes(Scope *scope) { return 0; }

	// supporting function for the type inference procedure that subclasses to provide implementation for
	// Note that for some expression such as logical-expression and range-expression the type is fixed
	// and the their sub-expression's expected types are already known. These expressions should apply
	// type inference of sub-expression as part of the resolveExprTypes(scope) function. They do not need
	// to extend this function.
	virtual int inferExprTypes(Scope *scope, Type *assignedType) { return 0; }

	// function to print and count any scope, type, and other semantic errors in an expression
	virtual int emitSemanticErrors(Scope *scope) { return 0; }
	
  public:
	//-------------------------------------------------------------------- Helper functions for Static Analysis
	
	// Following two functions are used to determine what version number to be used for epoch dependent
        // variables in any particular expression. They also update the epoch counter in the LPS referene if
        // an epoch dependent variable is a task-global data structure. Sub-classes should provide override
	// for the second function as needed.
        void analyseEpochDependencies(Space *space) { setEpochVersions(space, 0); }
        virtual void setEpochVersions(Space *space, int epoch) {}
	
	//------------------------------------------------------------- Common helper functions for Code Generation

	void setType(Type *type) { this->type = type; }

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	virtual void generateCode(std::ostringstream &stream, int indentLevel, Space *space = NULL);
        virtual void translate(std::ostringstream &stream,
                        int indentLevel, 
			int currentLineLength = 0, 
			Space *space = NULL);
};

class IntConstant : public Expr {
  protected:
    	int value;
	IntSize size;
  public:
    	IntConstant(yyltype loc, int val);
    	IntConstant(yyltype loc, int val, IntSize size);
    	const char *GetPrintNameForNode() { return "Integer-Constant"; }
    	void PrintChildren(int indentLevel);
	int getValue() { return value; }  

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new IntConstant(*GetLocation(), value, size); }
	ExprTypeId getExprTypeId() { return INT_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class FloatConstant : public Expr {
  protected:
    	float value;
  public:
    	FloatConstant(yyltype loc, float val);
    	const char *GetPrintNameForNode() { return "Float-Constant"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new FloatConstant(*GetLocation(), value); }
	ExprTypeId getExprTypeId() { return FLOAT_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class DoubleConstant : public Expr {
  protected:
    	double value;
  public:
    	DoubleConstant(yyltype loc, double val);
    	const char *GetPrintNameForNode() { return "Double-Constant"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new DoubleConstant(*GetLocation(), value); }
	ExprTypeId getExprTypeId() { return DOUBLE_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class BoolConstant : public Expr {
  protected:
    	bool value;
  public:
    	BoolConstant(yyltype loc, bool val);
    	const char *GetPrintNameForNode() { return "Boolean-Constant"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new BoolConstant(*GetLocation(), value); }
	ExprTypeId getExprTypeId() { return BOOL_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class StringConstant : public Expr {
  protected:
    	const char *value;
  public:
    	StringConstant(yyltype loc, const char *val);
    	const char *GetPrintNameForNode() { return "String-Constant"; }
    	void PrintChildren(int indentLevel);
	const char *getValue() { return value; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new StringConstant(*GetLocation(), strdup(value)); }
	ExprTypeId getExprTypeId() { return STRING_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class CharConstant : public Expr {
  protected:
    	char value;
  public:
    	CharConstant(yyltype loc, char val);
    	const char *GetPrintNameForNode() { return "Character-Constant"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone() { return new CharConstant(*GetLocation(), value); }
	ExprTypeId getExprTypeId() { return CHAR_CONST; };

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class ReductionVar : public Expr {
  protected:
	char spaceId;
	const char *name;

	// a field representation of the variable used in reduction for aiding semantic analysis
	FieldAccess *fieldRepresentation;
  public:
	ReductionVar(char spaceId, const char *name, yyltype loc);	
    	const char *GetPrintNameForNode() { return "Reduction-Var"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	const char *getName() { return name; }
        Node *clone() { return new ReductionVar(spaceId, strdup(name), *GetLocation()); }
	ExprTypeId getExprTypeId() { return REDUCTION_VAR; };
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
        const char *getBaseVarName() { return name; }
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

	char getSpaceId() { return spaceId; }

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &s, int i, int c, Space *space);
};

class ArithmaticExpr : public Expr {
  protected:
	Expr *left;
	ArithmaticOperator op;
	Expr *right;
  public:
	ArithmaticExpr(Expr *left, ArithmaticOperator op, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "Arithmatic-Expr"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return ARITH_EXPR; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int inferExprTypes(Scope *scope, Type *assignedType);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
};

class LogicalExpr : public Expr {
  protected:
	Expr *left;
	LogicalOperator op;
	Expr *right;
  public:
	LogicalExpr(Expr *left, LogicalOperator op, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "Logical-Expr"; }
    	void PrintChildren(int indentLevel);
	Expr *getLeft() { return left; }
	LogicalOperator getOp() { return op; }	
	Expr *getRight() { return right; }    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return LOGIC_EXPR; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        List<LogicalExpr*> *getANDBreakDown();
        static List<LogicalExpr*> *getIndexRestrictExpr(List<LogicalExpr*> *exprList,
                        std::ostringstream &stream,
                        const char *indexVar, const char *rangeExpr,
                        int indentLevel, Space *space,
                        bool xformedArrayRange, const char *arrayName, int dimensionNo);
        int isLoopRestrictExpr(const char *loopIndex);
        static bool transformIndexRestriction(std::ostringstream &stream,
                        const char *varName, const char *arrayName, int dimensionNo,
                        int indentLevel, Space *space,
                        bool normalizedToMinOfRange, bool lowerBound);
};

class EpochExpr : public Expr {
  protected:
	Expr *root;
	int lag;
  public:
	EpochExpr(Expr *root, int lag);
	const char *GetPrintNameForNode() { return "Epoch-Expr"; }
    	void PrintChildren(int indentLevel);
	Expr *getRootExpr() { return root; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return EPOCH_EXPR; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int inferExprTypes(Scope *scope, Type *assignedType);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
        const char *getBaseVarName() { return root->getBaseVarName(); }
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
};

class FieldAccess : public Expr {
  protected:
	Expr *base;
	Identifier *field;

	// this attributes tells if the field access should be treated as accessing via a reference or by
	// value. The default is access-by-value
	bool referenceField;

	// this attributes tell if the field access is an array reference and if YES then the dimensionality
	// of the array
	bool arrayField;
	int arrayDimensions; 

	// If the field is epoch dependent then we need to know which particular version of the field to be 
	// used during generating code for this field access. The default version number is zero. 
        int epochVersion;

	//----------------------------------------------------------------------- Code Generation Helper Properties
	
	// Two boolean variables are retained to determine if the field access is corresponding to an
        // array then whether its data or content is been accessed by the expression which this access
        // is a part of. When the access is indeed correspond to an array, we in addition need to know 
        // if local or global version (differ in how indexes are accessed) of the array been used.
        // These facts are important for the backend compiler to locate appropriate data structure for
        // the access as metadata and data for an array are kept separate. 
        bool metadata;
        bool local;

        // A flag to indicate if this field is an index access on an array where the index is part of 
        // some loop range traversal. In that case, we can just replace the field name with some other
        // back-end variable that holds the result of computation of multi to unidirectional index 
        // transform
        bool index;

  public:
	FieldAccess(Expr *base, Identifier *field, yyltype loc);	
	const char *GetPrintNameForNode() { return "Field-Access"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	Identifier *getField() { return field; }
	bool isTerminalField() { return base == NULL; }	    	
	ExprTypeId getExprTypeId() { return FIELD_ACC; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	void flagAsReferenceField() { referenceField = true; }
	void flagAsArrayField(int arrayDimensions);

	// This returns the first field in a chain of field accesses, e.g, if a.b.c.d is an expression then 
	// this will return 'a'; this does not work for accessing properties from elements of an array, i.e.,
	// if the access is like a[i].b then the function should return NULL.
	FieldAccess *getTerminalField();
	
	int resolveExprTypes(Scope *scope);
	int inferExprTypes(Scope *scope, Type *assignedType);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
        const char *getBaseVarName();
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);
	void setEpochVersion(int epoch) { this->epochVersion = epoch; }
	
	//------------------------------------------------------------- Common helper functions for Code Generation

	Expr *getBase() { return base; }
        void setMetadata(bool metadata) { this->metadata = metadata; }
        bool isMetadata() { return metadata; }
	void markLocal() { local = true; }
        bool isLocal() { return local; }
	void markAsIndex() { index = true; }
        bool isIndex() { return index; }
	bool isLocalTerminalField();	

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
        void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        void translateIndex(std::ostringstream &stream, const char *array, int dimension);
        bool isEnvArrayDim();
        void translateEnvArrayDim(std::ostringstream &stream,
                        int indentLevel,
                        int currentLineLength, Space *space);
};

class RangeExpr : public Expr {
  protected:
	FieldAccess *index;
	Expr *range;
	Expr *step;
	bool loopingRange;
  public:
	// constructor to be used when the range expression is a part of a for loop
	RangeExpr(Identifier *index, Expr *range, Expr *step, yyltype loc);
	// constructor to be used when it is not part of a for loop		
	RangeExpr(Expr *index, Expr *range, yyltype loc);	
	
	const char *GetPrintNameForNode() { return "Range-Expr"; }
    	void PrintChildren(int indentLevel);	    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return RANGE_EXPR; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//------------------------------------------------------------- Common helper functions for Code Generation

	FieldAccess *getIndex() { return index; }
	Expr *getRange() { return range; }
	Expr *getStep() { return step; }

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	const char *getIndexExpr();
        const char *getRangeExpr(Space *space);
        const char *getStepExpr(Space *space);
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        const char *getBaseArrayForRange(Space *executionSpace);
        int getDimensionForRange(Space *executionSpace);
        void generateLoopForRangeExpr(std::ostringstream &stream,
                        int indentation, Space *space, const char *loopbounRestrictCond = NULL);
        void translateArrayRangeExprCheck(std::ostringstream &stream, int indentLevel, Space *space);
        void generateAssignmentExprForXformedIndex(std::ostringstream &stream,
                        int indentLevel, Space *space);
};	

class AssignmentExpr : public Expr {
  protected:
	Expr *left;
	Expr *right;
  public:
	AssignmentExpr(Expr *left, Expr *right, yyltype loc);	
	const char *GetPrintNameForNode() { return "Assignment-Expr"; }
    	void PrintChildren(int indentLevel);
	Expr *getLeft() { return left; }
	Expr *getRight() { return right; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return ASSIGN_EXPR; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int inferExprTypes(Scope *scope, Type *assignedType);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class IndexRange : public Expr {
  protected:
	Expr *begin;
	Expr *end;
	bool fullRange;
	
	// An index-range can be used to denote a parts of an array. It can also be used as a 
	// range variable for repeat iteration control. This variable indicates which use is
	// intended in a particular instance.
	bool partOfArray;
  public:
	IndexRange(Expr *begin, Expr *end, bool partOfArray, yyltype loc);
	const char *GetPrintNameForNode() { return "Index-Range"; }
    	void PrintChildren(int indentLevel);
	bool isFullRange() { return fullRange; }	    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return INDEX_RANGE; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
	Expr *getBegin() { return begin; }
	Expr *getEnd() { return end; }

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;

	// This variable serves as a flag to avoid repititive transformation of compute stage parameters
	// during the resolution of compute stages based on invocation arguments. If true, this means
	// the current index access of the base array represented by this expression has been added 
	// into the array access expression chain during some earlier step of the transformation process.
	// So the index should be skipped any further transformation.
	bool fillerIndexAccess;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "Array-Access"; }
    	void PrintChildren(int indentLevel);
	Expr *getBase() { return base; }
	Expr *getIndex() { return index; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return ARRAY_ACC; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	void flagAsFillerIndexAccess() { fillerIndexAccess = true; }
	
	// tells which dimension of the array this expression is accessing
	int getIndexPosition();

	// These two functions are needed to apply array-access expression translations as part of the
	// polymorphic compute stage resolution process. We generate a new access expression chain 
	// relative to the original array from an access to an argument array representing a part of the 
	// original. For example, a row or column of a matrix. We set up new base and index expressions 
	// to refer to the transformed array access expression chain through these two pointers.
	void setBase(Expr *baseExpr) { this->base = baseExpr; }
	void setIndex(Expr *indexExpr) { this->index = indexExpr; }
        
	// this returns the base array, field access, expression the array access has originated from
	Expr *getEndpointOfArrayAccess();

	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
        const char *getBaseVarName() { return base->getBaseVarName(); }

	// tells if the current array access expression is the last index access in the chain of array
	// indexes beginning at the base array
	bool isFinalIndexAccess();

	// this checks if an index variable has been used to access the current dimension of the array 
	// whether or not the same index variable has been used to access any earlier dimension
	bool earlierIndexAssociationExists(const char *indexName);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

        void generate1DIndexAccess(std::ostringstream &stream, int indentLevel,
                        const char *array, ArrayType *type, Space *space);
        void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        void generateXformedIndex(std::ostringstream &stream, int indentLevel,
                        const char *indexExpr,
                        const char *arrayName, int dimensionNo, Space *space);
};

class FunctionCall : public Expr {
  protected:
	Identifier *base;
	List<Expr*> *arguments;
  public:
	FunctionCall(Identifier *base, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "Function-Call"; }
    	void PrintChildren(int indentLevel);	    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return FN_CALL; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
};

class NamedArgument : public Node {
  protected:
	const char *argName;
	Expr *argValue;
  public:
	NamedArgument(char *argName, Expr *argValue, yyltype loc);	
	const char *GetPrintNameForNode() { return "Named-Argument"; }
    	void PrintChildren(int indentLevel);	    	
	const char *getName() { return argName; }
	Expr *getValue() { return argValue; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/

	void generateAssignment(Expr *object, std::ostringstream &stream, int indentLevel);
};

class NamedMultiArgument : public Node {
  protected:
	const char *argName;
	List<Expr*> *argList;
  public:
	NamedMultiArgument(char *argName, List<Expr*> *argList, yyltype loc);
	const char *GetPrintNameForNode() { return "Named-Multi-Argument"; }
    	void PrintChildren(int indentLevel);	    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	const char *getName() { return argName; }
	List<Expr*> *getArgList() { return argList; }
};

class TaskInvocation : public Expr {
  protected:
	List<NamedMultiArgument*> *invocationArgs;

	// a reference to the task definition generated during scope-and-type checking is retained to aid
	// code generation
	TaskDef *taskDef;	
  public:
	TaskInvocation(List<NamedMultiArgument*> *invocationArgs, yyltype loc);
	const char *GetPrintNameForNode() { return "Task-Invocation"; }
    	void PrintChildren(int indentLevel);	    	

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return TASK_INVOKE; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	int emitSemanticErrors(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);

	// helper functions to retrieve different types of task invocation arguments 
	const char *getTaskName();
	FieldAccess *getEnvArgument();
	List<Expr*> *getInitArguments();
	List<Expr*> *getPartitionArguments();

  protected:
	NamedMultiArgument *retrieveArgByName(const char *argName);

  public:
	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
        
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class ObjectCreate : public Expr {
  protected:
	Type *objectType;
	List<NamedArgument*> *initArgs;
  public:
	ObjectCreate(Type *objectType, List<NamedArgument*> *initArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "Object-Create"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return OBJ_CREATE; };
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypes(Scope *scope);
	void retrieveTerminalFieldAccesses(List<FieldAccess*> *fieldList);
	void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis
        
	void setEpochVersions(Space *space, int epoch);

	//-------------------------------------------------------------------------- Code Generation Hack Functions
        /**********************************************************************************************************
          The code generation related function definitions that are placed here are platform specific. So ideally 
          they should not be included here and the frontend compiler should be oblivious of them. However, as we
          ran out of time in overhauling the old compilers, instead of redesigning the code generation process, we 
          decided to keep the union of old function definitions in the frontend and put their implementations in
          relevent backend compilers.   
        **********************************************************************************************************/
	
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
        void generateCodeForProperties(Expr *object, std::ostringstream &stream, int indentLevel);
        static bool isDynamicArrayCreate(Expr *candidateExpr);
};

#endif
