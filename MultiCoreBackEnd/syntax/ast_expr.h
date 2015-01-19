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

	// Helper functions for semantic analysis
	void performTypeInference(Scope *executionScope);
        void checkSemantics(Scope *executionScope, bool ignoreTypeFailures) {
		resolveType(executionScope, ignoreTypeFailures);
	}
	virtual void resolveType(Scope *scope, bool ignoreFailure) {}
	virtual void inferType(Scope *scope, Type *rootType) {}   
	void inferType(Scope *scope) { inferType(scope, type); }   
	Type *getType() { return type; }
	
	// Helper functions for static analysis
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(
			TaskGlobalReferences *globalReferences) {
		return new Hashtable<VariableAccess*>;
	}
	virtual const char *getBaseVarName() { return NULL; }
	
	// Helper functions for code generation
	virtual void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
	virtual void translate(std::ostringstream &stream, 
			int indentLevel, int currentLineLength);
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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
	void translate(std::ostringstream &s, int i, int c) { s << value; }
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

	// for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	
	// for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);

	// for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
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
	
	// for semantic analysis
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	// for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
	List<FieldAccess*> *getTerminalFieldAccesses();
	// Local expressions sometimes are added to indexed based parallel loop statement blocks to further
	// restrict the range of indexes been traversed by the for loop. Given that there might be multiple
	// nested for loops in the target code correspond to a single loop in IT source, we need to know
	// what loop is the best place for a restricting logical expression to put into and then do that.
	// For this to be done efficiently we may need to break restricting conditions connected by AND op-
	// erators and put different parts in different location. So the following method has been added to
	// break a collective of AND statements into a list of such statements
	List<LogicalExpr*> *getANDBreakDown(); 
};

class ReductionExpr : public Expr {
  protected:
	ReductionOperator op;
	Expr *right;
  public:
	ReductionExpr(char *opName, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "ReductionExpr"; }
    	void PrintChildren(int indentLevel);

	// for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	// for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);

	// for code generation
	List<FieldAccess*> *getTerminalFieldAccesses();
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength) { stream << "\"reduction\""; }
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

	// for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Expr *getRootExpr() { return root; }

	// for code generation
	const char *getBaseVarName() { return root->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<FieldAccess*> *getTerminalFieldAccesses();
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength) { stream << "\"epoch-expr\""; }
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
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);   
	const char *getBaseVarName();
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// an additional helper function for static analysis
	bool isTerminalField() { return base == NULL; }
	
	// helper functions for back end compiler
	bool isLocalTerminalField();
	void markLocal() { local = true; }
	void setMetadata(bool metadata) { this->metadata = metadata; }
	bool isLocal() { return local; }
	bool isMetadata() { return metadata; }
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
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
	void resolveType(Scope *scope, bool ignoreFailure);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	const char *getIndexName() { return index->getName(); }
	
	// helper functions for backend compiler
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
	// As a range expression can be used as the condition for a repeat loop that needs to be
	// translated a for loop with other content inside, it provides following functions so that
	// the caller can get string equivalent of its parts and generate the loop
	const char *getIndexExpr();
	const char *getRangeExpr();
	const char *getStepExpr();
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
	// of translation of the range expression
	void generateLoopForRangeExpr(std::ostringstream &stream, int indentation, Space *space);
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

	// for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);

	// for static analysis
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	const char *getBaseVarName() { return left->getBaseVarName(); }
	
	// for code generation
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
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
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength) { stream << "\"subrange\""; }
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "ArrayAccess"; }
    	void PrintChildren(int indentLevel);

	// for semantic analysis	    	
	void resolveType(Scope *scope, bool ignoreFailure);

	// for static analysis
	const char *getBaseVarName() { return base->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	int getIndexPosition();
	
	// for code generation
	List<FieldAccess*> *getTerminalFieldAccesses();
	Expr *getEndpointOfArrayAccess();
	void generate1DIndexAccess(std::ostringstream &stream, const char *array, ArrayType *type);
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength);
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
	void translate(std::ostringstream &stream, int indentLevel, int currentLineLength) { stream << "\"function-call\""; }
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
};

class TaskInvocation : public Expr {
  protected:
	Identifier *taskName;
	Identifier *environment;
	List<OptionalInvocationParams*> *optionalArgs;	
  public:
	TaskInvocation(Identifier *taskName, Identifier *environment, 
		List<OptionalInvocationParams*> *optionalArgs, yyltype loc);
	const char *GetPrintNameForNode() { return "TaskInvocation"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
};

class ObjectCreate : public Expr {
  protected:
	Type *objectType;
	List<Expr*> *initArgs;
  public:
	ObjectCreate(Type *objectType, List<Expr*> *initArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "ObjectCreate"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
};

#endif
