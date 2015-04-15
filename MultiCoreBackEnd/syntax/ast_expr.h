/* File: ast_expr.h
 * ----------------
 * The Expr class and its subclasses are used to represent
 * expressions in the parse tree.  For each expression in the
 * language (add, call, New, etc.) there is a corresponding
 * node class for that construct. 
 */


#ifndef _H_ast_expr
#define _H_ast_expr

#include "ast.h"
#include "ast_stmt.h"
#include "ast_type.h"
#include "ast_task.h"

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../semantics/scope.h"
#include "../static-analysis/data_access.h"

#include <sstream>

enum ArithmaticOperator { ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, LEFT_SHIFT, RIGHT_SHIFT, POWER };
enum LogicalOperator { AND, OR, NOT, EQ, NE, GT, LT, GTE, LTE };
enum ReductionOperator { SUM, PRODUCT, MAX, MIN, AVG, MAX_ENTRY, MIN_ENTRY };

class TaskDef;
class FieldAccess;
class Space;

class Expr : public Stmt {
  protected:
	Type *type;
  public:
    	Expr(yyltype loc) : Stmt(loc) { type = NULL; }
    	Expr() : Stmt() { type = NULL; }

	//--------------------------------------------------------------------Helper functions for semantic analysis
	void performTypeInference(Scope *executionScope);
        void checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
		resolveType(executionScope, ignoreTypeFailures);
	}
	// As IT uses static typing with type inference, a single pass over the abstract syntax tree is not
	// sufficient to determine the types of all variables. An impressive solution would be to do recursive
	// type resolving until no new type can be resolved for IT. We did a simpler solution where, we first
	// try to resolve type and ignore any failure in the resolution process. Then we try to infer types of
	// variables with unresolved types using information from varibles with already known types. Finally
	// we do another type resolving and this time through errors if we some variables' types still remain
	// unresolved.
	virtual void resolveType(Scope *scope, bool ignoreFailure) {}
	virtual void inferType(Scope *scope, Type *rootType) {}   
	void inferType(Scope *scope) { inferType(scope, type); }   
	Type *getType() { return type; }
	
	//-----------------------------------------------------------------------Helper functions for static analysis
	// This function decides, as its name suggests, the global variables been accessed by the expression.
	// It can track assignment of global array reference assignments to some local variables and then 
	// indirect changes to the global array through that local reference. This analysis is required to 
	// determine the execution order, communication, and synchronization dependencies among compute stages.
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(
			TaskGlobalReferences *globalReferences) {
		return new Hashtable<VariableAccess*>;
	}
	// This function finds out the root object within which an element been accessed or modified by some
	// expression. It makes sense only for array-access and field-access type expression. The function is
	// however added to the common expression class to simply some recursive array/field access finding
 	// process.
	virtual const char *getBaseVarName() { return NULL; }
	
	//-----------------------------------------------------------------------Helper functions for code generation
	virtual void generateCode(std::ostringstream &stream, int indentLevel, Space *space = NULL);
	virtual void translate(std::ostringstream &stream, 
			int indentLevel, int currentLineLength = 0, Space *space = NULL);
	virtual List<FieldAccess*> *getTerminalFieldAccesses();
	static void copyNewFields(List<FieldAccess*> *destination, List<FieldAccess*> *source);
	void setType(Type *type) { this->type = type; }
};

class IntConstant : public Expr {
  protected:
    	int value;
  public:
    	IntConstant(yyltype loc, int val);
    	const char *GetPrintNameForNode() { return "IntConstant"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::intType; }
	void inferType(Scope *scope, Type *rootType);
	int getValue() { return value; }  
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }
};

class FloatConstant : public Expr {
  protected:
    	float value;
  public:
    	FloatConstant(yyltype loc, float val);
    	const char *GetPrintNameForNode() { return "FloatConstant"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::floatType; }
	void inferType(Scope *scope, Type *rootType);   
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }
};

class DoubleConstant : public Expr {
  protected:
    	double value;
  public:
    	DoubleConstant(yyltype loc, double val);
    	const char *GetPrintNameForNode() { return "DoubleConstant"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::doubleType; }
	void inferType(Scope *scope, Type *rootType);   
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }
};

class BoolConstant : public Expr {
  protected:
    	bool value;
  public:
    	BoolConstant(yyltype loc, bool val);
    	const char *GetPrintNameForNode() { return "BoolConstant"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::boolType; }
	void inferType(Scope *scope, Type *rootType);   
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }
};

class StringConstant : public Expr {
  protected:
    	const char *value;
  public:
    	StringConstant(yyltype loc, const char *val);
    	const char *GetPrintNameForNode() { return "StringConstant"; }
    	void PrintChildren(int indentLevel);
	const char *getValue() { return value; }
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::stringType; }
	void inferType(Scope *scope, Type *rootType);   
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << '\"' << value << '\"'; }
};

class CharacterConstant : public Expr {
  protected:
    	char value;
  public:
    	CharacterConstant(yyltype loc, char val);
    	const char *GetPrintNameForNode() { return "CharacterConstant"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure) { type = Type::charType; }
	void inferType(Scope *scope, Type *rootType);   
	void translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }
};

class ArithmaticExpr : public Expr {
  protected:
	Expr *left;
	ArithmaticOperator op;
	Expr *right;
  public:
	ArithmaticExpr(Expr *left, ArithmaticOperator op, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "ArithmaticExpr"; }
    	void PrintChildren(int indentLevel);

	//--------------------------------------------------------------------------------------for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	
	//----------------------------------------------------------------------------------------for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);

	//----------------------------------------------------------------------------------------for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	List<FieldAccess*> *getTerminalFieldAccesses();
};

class LogicalExpr : public Expr {
  protected:
	Expr *left;
	LogicalOperator op;
	Expr *right;
  public:
	LogicalExpr(Expr *left, LogicalOperator op, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "LogicalExpr"; }
    	void PrintChildren(int indentLevel);
	Expr *getLeft() { return left; }
	LogicalOperator getOp() { return op; }	
	Expr *getRight() { return right; }    	
	
	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	//----------------------------------------------------------------------------------------for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//----------------------------------------------------------------------------------------for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	List<FieldAccess*> *getTerminalFieldAccesses();
	
	// Local expressions sometimes are added to indexed based parallel loop statement blocks to further
	// restrict the range of indexes been traversed by the for loop. Given that there might be multiple
	// nested for loops in the target code correspond to a single loop in IT source, we need to know
	// what loop is the best place for a restricting logical expression to put into and then do that.
	// For this to be done efficiently we may need to break restricting conditions connected by AND op-
	// erators and put different parts in different location. So the following method has been added to
	// break a collective of AND statements into a list of such statements
	List<LogicalExpr*> *getANDBreakDown(); 

	// This function generates, as its name suggests, a starting or ending condition for an index from 
	// logical expressions that are used as restricting conditions in parallel loops based on that index. 
	// An example of such restriction can be "do { ... } for k in matrix AND k > expr." Here we will like
	// to begin the translated C++ loops to start from "k = expr + 1." On the other hand, if the range
	// to be traversed by the index is a decreasing range than the same condition should be used to exit
	// from the loop instead of as a starting condition. If none of the expressions in the list can be used 
	// to restrict generated index loop this way due to the nature of those expressions, then it does 
	// nothing.
	// The last three parameters are for determining if the index under concern traverses a reordered
	// array dimension, and if it does then transform the index start or end restriction that may be 
	// applied to the added restrictions.
	// The function returns a filtered list of index restriction expressions if some of the expressions in
	// the original list can be successfully and precisely applied on the loop. Otherwise, it returns the
	// original list      
	static List<LogicalExpr*> *getIndexRestrictExpr(List<LogicalExpr*> *exprList, 
			std::ostringstream &stream, 
			const char *indexVar, const char *rangeExpr, 
			int indentLevel, Space *space,
			bool xformedArrayRange, const char *arrayName, int dimensionNo);
	
	// This is a supporting function for the function above to determine whether to consider of skip an
	// expression. Instead of a boolean value, it returns an integer as we need to know on which side of
	// the expression the index variable lies. So it returns -1 if the expression is a loop restrict
	// condition and the loop index is on the right of the expression, 1 if the loop index is on the left,
	// and 0 if the expression is not a loop restrict condition.
	int isLoopRestrictExpr(const char *loopIndex);
	
	// This function transforms a variable holding the value of an index restricting expression based on
	// the partitioning of the array dimension that the index is traversing -- when the loop corresponds to
	// a reordered array dimension traversal, of course. Notice the second last parameter of this function. 
	// This is used to determine what to set the value of the variable to if it falls outside the range of 
	// the dimension that falls within the LPU where the generated code will execute. This is needed as this 
	// function is used in context where we do not know if the variable is within the boundary of the LPU.
	// The last parameter is used to determine if a lower bound or an upper bound should be attempted by 
	// the transformation process when the index is not in within the boundary of the LPU and we are sured
	// about its position relative to the LPU boundary. TODO probably we can exclude the second last parameter
	// if we do some refactoring in the implementation. Varify the correctness of the new implementation if
	// you attempt that.
	// The function returns a boolean value indicating if it made a precise transformation of the given 
	// restriction or not.   
	static bool transformIndexRestriction(std::ostringstream &stream, 
			const char *varName, const char *arrayName, int dimensionNo, 
			int indentLevel, Space *space, 
			bool normalizedToMinOfRange, bool lowerBound);
};

class ReductionExpr : public Expr {
  protected:
	ReductionOperator op;
	Expr *right;
	// a reference to the loop that contains this reduction operation
	LoopStmt *reductionLoop;
  public:
	ReductionExpr(char *opName, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "ReductionExpr"; }
    	void PrintChildren(int indentLevel);

	//--------------------------------------------------------------------------------------for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	//----------------------------------------------------------------------------------------for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);

	//----------------------------------------------------------------------------------------for code generation
	List<FieldAccess*> *getTerminalFieldAccesses();
	// the following function should be used to assign the result of reduction to the left hand side
	// of the assignment expression the reduction is a part of
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	// the following function should be used to generate the code for doing the actual reduction
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
	// the following function should be used to initialize the reduction loop (declaration of supplementary
	// variables, setting initial value for result variable)
	void setupForReduction(std::ostringstream &stream, int indentLevel);
	// the following function should be used to do any finalization, if needed, after reduction loop completes
	void finalizeReduction(std::ostringstream &stream, int indentLevel);
};

class EpochValue : public Expr {
  protected:
	Identifier *epoch;
	int lag;
  public:
	EpochValue(Identifier *epoch, int lag);	
	const char *GetPrintNameForNode() { return "EpochValue"; }
    	void PrintChildren(int indentLevel);
	void resolveType(Scope *scope, bool ignoreFailure);
	Identifier *getId() { return epoch; }
};

class EpochExpr : public Expr {
  protected:
	Expr *root;
	EpochValue *epoch;
  public:
	EpochExpr(Expr *root, EpochValue *epoch);
	const char *GetPrintNameForNode() { return "EpochExpr"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------------------------for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Expr *getRootExpr() { return root; }

	//--------------------------------------------------------------------------------------for code generation
	const char *getBaseVarName() { return root->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<FieldAccess*> *getTerminalFieldAccesses();
	void translate(std::ostringstream &stream, int indentLevel, 
			int currentLineLength, Space *space) { stream << "\"epoch-expr\""; }
};

class FieldAccess : public Expr {
  protected:
	Expr *base;
	Identifier *field;
	// Two boolean variables are retained to determine if the field access is corresponding to an
	// array then whether its data or content is been accessed by the expression which this access
	// is a part of. When the access is indeed correspond to an array, we in addition need to know 
	// if local or global version (differ in how indexes are accessed) of the array been used.
	// These facts are important for the backend compiler to locate appropriate data structure for
	// the access as metadata and data for an array are kept separate. 
	bool metadata;
	bool local;
	// a flag to indicate if this field is an index access on an array where the index is part of 
	// some loop range traversal. In that case, we can just replace the field name with some other
	// back-end variable that holds the result of computation of multi to unidirectional index 
	// transform
	bool index; 
  public:
	FieldAccess(Expr *base, Identifier *field, yyltype loc);	
	const char *GetPrintNameForNode() { return "FieldAccess"; }
    	void PrintChildren(int indentLevel);	    	
	
	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);   
	
	//----------------------------------------------------------------------------------------for static analysis
	const char *getBaseVarName();
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	bool isTerminalField() { return base == NULL; }
	
	//---------------------------------------------------------------------helper functions for back end compiler
	bool isLocalTerminalField();
	void markLocal() { local = true; }
	void setMetadata(bool metadata) { this->metadata = metadata; }
	bool isLocal() { return local; }
	bool isMetadata() { return metadata; }
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	Expr *getBase() { return base; }
	Identifier *getField() { return field; }
	bool isEqual(FieldAccess *other);
	List<FieldAccess*> *getTerminalFieldAccesses();
	void markAsIndex() { index = true; }
	bool isIndex() { return index; }
	
	// if the field access is an array index access where the index correspond to some loop iteration
	// index then the actual memory location for the index is available in a generated back-end 
	// variable. This method writes the name of that variable on the stream as a replacement of source
	// index
	void translateIndex(std::ostringstream &stream, const char *array, int dimension);
};

class RangeExpr : public Expr {
  protected:
	Identifier *index;
	Expr *range;
	Expr *step;
	bool loopingRange;
	// a variable corresponding to index to work as a holder of index transformation information
	// during range checking, if need, during code generation 
	FieldAccess *indexField;
  public:
	RangeExpr(Identifier *index, Expr *range, Expr *step, bool loopingRange, yyltype loc);		
	const char *GetPrintNameForNode() { return "RangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
	
	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	const char *getIndexName() { return index->getName(); }
	
	//----------------------------------------------------------------------------------------for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	
	// As a range expression can be used as the condition for a repeat loop that needs to be
	// translated a for loop with other content inside, it provides following functions so that
	// the caller can get string equivalent of its parts and generate the loop
	const char *getIndexExpr();
	const char *getRangeExpr(Space *space);
	const char *getStepExpr(Space *space);
	List<FieldAccess*> *getTerminalFieldAccesses();

	// As the range within a range expression may be the dimension of some array, there might be a
	// need for index transformation depending on whether or not that dimension is been reordered 
	// by the partition function. Therefore, the following two methods are provided to identify the
	// array and dimension no correspond to the range expression, if applicable. At code generation
	// this information will be used to determine if any adjustment is needed in the index before
	// it can be used for anything else. TODO note that a normal field access within any other expr
	// may involve accessing the dimension range of some array. Nonetheless, we do not provide methods
	// similar to this in the field-access and other classes as it does not make sense accessing the
	// min and max of an array dimension in a computation if the underlying partition function for
	// the LPS can reorder data. If, however, in the future such accesses seem to be normal then we
	// need to elaborate on this logic. 
	const char *getBaseArrayForRange(Space *executionSpace);
	int getDimensionForRange(Space *executionSpace);
	
	// To generate a for loop without the closing parenthesis if we need a standard implementation
	// of translation of the range expression. The last parameter is used by the caller to pass any
	// additional restriction to be applied to the start and/or end condition of the loop that are
	// generated by the range expression by default
	void generateLoopForRangeExpr(std::ostringstream &stream, 
			int indentation, Space *space, const char *loopbounRestrictCond = NULL);
	
	// This function generates an accurate index inclusion check when the range in this expression
	// correspond to a reordered index of some array.
	void translateArrayRangeExprCheck(std::ostringstream &stream, int indentLevel, Space *space);
	
	// This function generate an assignment expression for actual index when the range expression
	// results in a traversal of a partition range of a reordered dimension of an array
	void generateAssignmentExprForXformedIndex(std::ostringstream &stream, 
			int indentLevel, Space *space);
};

class SubpartitionRangeExpr : public Expr {
  protected:
	char spaceId;
  public:
	SubpartitionRangeExpr(char spaceId, yyltype loc);
	const char *GetPrintNameForNode() { return "SubpartitionRangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope) { type = Type::boolType; }
	char getSpaceId() { return spaceId; }
};

class AssignmentExpr : public Expr {
  protected:
	Expr *left;
	Expr *right;
  public:
	AssignmentExpr(Expr *left, Expr *right, yyltype loc);	
	const char *GetPrintNameForNode() { return "AssignmentExpr"; }
    	void PrintChildren(int indentLevel);
	Expr *getLeft() { return left; }
	Expr *getRight() { return right; }

	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	//----------------------------------------------------------------------------------------for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	const char *getBaseVarName() { return left->getBaseVarName(); }
	
	//----------------------------------------------------------------------------------------for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	List<FieldAccess*> *getTerminalFieldAccesses();
	
	// Assignment expression overrides generate-code function along with common translate function
	// to break compound assignment statements to multiple simple ones. Also we currently support
	// direct assignment of multiple dimensions from one array to another. That need to be tackled
	// by overriding generate-code. TODO later we have to find out a better way of doing this or 
	// rethink that array assignment feature all together.
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class SubRangeExpr : public Expr {
  protected:
	Expr *begin;
	Expr *end;
	bool fullRange;
  public:
	SubRangeExpr(Expr *begin, Expr *end, yyltype loc);
	const char *GetPrintNameForNode() { return "SubRangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
	void inferType(Scope *scope, Type *rootType); 
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<FieldAccess*> *getTerminalFieldAccesses();
	bool isFullRange() { return fullRange; }
	Expr *getBegin() { return begin; }	
	Expr *getEnd() { return end; }
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "ArrayAccess"; }
    	void PrintChildren(int indentLevel);
	Expr *getBase() { return base; }
	Expr *getIndex() { return index; }

	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);

	//----------------------------------------------------------------------------------------for static analysis
	const char *getBaseVarName() { return base->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	int getIndexPosition();
	
	//----------------------------------------------------------------------------------------for code generation
	List<FieldAccess*> *getTerminalFieldAccesses();
	Expr *getEndpointOfArrayAccess();
	void generate1DIndexAccess(std::ostringstream &stream, 
			const char *array, ArrayType *type, Space *space);
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
};

class FunctionCall : public Expr {
  protected:
	Identifier *base;
	List<Expr*> *arguments;
  public:
	FunctionCall(Identifier *base, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "FunctionCall"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreType);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<FieldAccess*> *getTerminalFieldAccesses();
	void translate(std::ostringstream &stream, int indentLevel, 
			int currentLineLength, Space *space) { stream << "\"function-call\""; }
};

class OptionalInvocationParams : public Node {
  protected:
	Identifier *section;
	List<Expr*> *arguments;
  public:
	static const char *InitializeSection, *PartitionSection;
	OptionalInvocationParams(Identifier *section, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "OptionalParameters"; }
    	void PrintChildren(int indentLevel);	    	
	void validateTypes(Scope *scope, TaskDef *taskDef, bool ignoreFailure);
	bool isPartitionSection();
	List<Expr*> *getArguments() { return arguments; }
};

class TaskInvocation : public Expr {
  protected:
	Identifier *taskName;
	TaskDef *taskDef;
	Identifier *environment;
	List<OptionalInvocationParams*> *optionalArgs;	
  public:
	TaskInvocation(Identifier *taskName, Identifier *environment, 
		List<OptionalInvocationParams*> *optionalArgs, yyltype loc);
	const char *GetPrintNameForNode() { return "TaskInvocation"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class InitializerArg : public Node {
  protected:
	const char *argName;
	Expr *argValue;
  public:
	InitializerArg(char *argName, Expr *argValue, yyltype loc);	
	const char *GetPrintNameForNode() { return "InitializerArg"; }
    	void PrintChildren(int indentLevel);	    	
	void validateType(Scope *scope, TupleDef *objectDef, bool ignoreFailure);
	const char *getName() { return argName; }
	Expr *getValue() { return argValue; }
	void generateAssignment(Expr *object, std::ostringstream &stream, int indentLevel);
};

class ObjectCreate : public Expr {
  protected:
	Type *objectType;
	List<InitializerArg*> *initArgs;
  public:
	ObjectCreate(Type *objectType, List<InitializerArg*> *initArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "ObjectCreate"; }
    	void PrintChildren(int indentLevel);
	
	//--------------------------------------------------------------------------------------for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	
	//----------------------------------------------------------------------------------------for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space);
	void generateCodeForProperties(Expr *object, std::ostringstream &stream, int indentLevel);
	static bool isDynamicArrayCreate(Expr *candidateExpr);
};

#endif
