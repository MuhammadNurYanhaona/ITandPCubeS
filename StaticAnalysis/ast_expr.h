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

#include "list.h"
#include "hashtable.h"
#include "scope.h"
#include "data_access.h"

enum ArithmaticOperator { ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, LEFT_SHIFT, RIGHT_SHIFT, POWER };
enum LogicalOperator { AND, OR, NOT, EQ, NE, GT, LT, GTE, LTE };
enum ReductionOperator { SUM, PRODUCT, MAX, MIN, AVG, MAX_ENTRY, MIN_ENTRY };

class TaskDef;

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
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
		return new Hashtable<VariableAccess*>;
	}
	virtual const char *getBaseVarName() { return NULL; }
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
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
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
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
};

class ReductionExpr : public Expr {
  protected:
	ReductionOperator op;
	Expr *right;
  public:
	ReductionExpr(char *opName, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "ReductionExpr"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
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
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);
	Expr *getRootExpr() { return root; }
	const char *getBaseVarName() { return root->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
};

class FieldAccess : public Expr {
  protected:
	Expr *base;
	Identifier *field;
  public:
	FieldAccess(Expr *base, Identifier *field, yyltype loc);	
	const char *GetPrintNameForNode() { return "FieldAccess"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);   
	const char *getBaseVarName();
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	bool isTerminalField() { return base == NULL; }
};

class RangeExpr : public Expr {
  protected:
	Identifier *index;
	Expr *range;
	Expr *step;
	bool loopingRange; 
  public:
	RangeExpr(Identifier *index, Expr *range, Expr *step, bool loopingRange, yyltype loc);		
	const char *GetPrintNameForNode() { return "RangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
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
	void resolveType(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *rootType);   
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	const char *getBaseVarName() { return left->getBaseVarName(); }
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
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "ArrayAccess"; }
    	void PrintChildren(int indentLevel);	    	
	void resolveType(Scope *scope, bool ignoreFailure);
	const char *getBaseVarName() { return base->getBaseVarName(); }
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	int getIndexPosition();
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
