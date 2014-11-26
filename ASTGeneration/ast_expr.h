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
#include "list.h"

enum ArithmaticOperator { ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, LEFT_SHIFT, RIGHT_SHIFT, POWER };
enum LogicalOperator { AND, OR, NOT, EQ, NE, GT, LT, GTE, LTE };
enum ReductionOperator { SUM, PRODUCT, MAX, MIN, AVG, MAX_ENTRY, MIN_ENTRY };

class Expr : public Stmt {
  public:
    	Expr(yyltype loc) : Stmt(loc) {}
    	Expr() : Stmt() {}
};

class IntConstant : public Expr {
  protected:
    	int value;
  public:
    	IntConstant(yyltype loc, int val);
    	const char *GetPrintNameForNode() { return "IntConstant"; }
    	void PrintChildren(int indentLevel);
};

class FloatConstant : public Expr {
  protected:
    	float value;
  public:
    	FloatConstant(yyltype loc, float val);
    	const char *GetPrintNameForNode() { return "FloatConstant"; }
    	void PrintChildren(int indentLevel);
};

class DoubleConstant : public Expr {
  protected:
    	double value;
  public:
    	DoubleConstant(yyltype loc, double val);
    	const char *GetPrintNameForNode() { return "DoubleConstant"; }
    	void PrintChildren(int indentLevel);
};

class BoolConstant : public Expr {
  protected:
    	bool value;
  public:
    	BoolConstant(yyltype loc, bool val);
    	const char *GetPrintNameForNode() { return "BoolConstant"; }
    	void PrintChildren(int indentLevel);
};

class StringConstant : public Expr {
  protected:
    	char *value;
  public:
    	StringConstant(yyltype loc, const char *val);
    	const char *GetPrintNameForNode() { return "StringConstant"; }
    	void PrintChildren(int indentLevel);
};

class CharacterConstant : public Expr {
  protected:
    	char value;
  public:
    	CharacterConstant(yyltype loc, char val);
    	const char *GetPrintNameForNode() { return "CharacterConstant"; }
    	void PrintChildren(int indentLevel);
};

class IndexRangeExpr : public Expr {
  protected:
	Identifier *index;
	Identifier *array;
  public:
	IndexRangeExpr(Identifier *index, Identifier *array);
	const char *GetPrintNameForNode() { return "IndexRangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
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
};

class ReductionExpr : public Expr {
  protected:
	Expr *left;
	ReductionOperator op;
	Expr *right;
  public:
	ReductionExpr(Expr *left, ReductionOperator op, Expr *right, yyltype loc);
	const char *GetPrintNameForNode() { return "ReductionExpr"; }
    	void PrintChildren(int indentLevel);	    	
};

class EpochValue : public Expr {
  protected:
	Identifier *epoch;
	int lag;
  public:
	EpochValue(Identifier *epoch, int lag);	
	const char *GetPrintNameForNode() { return "EpochValue"; }
    	void PrintChildren(int indentLevel);	    	
};

class EpochExpr : public Expr {
  protected:
	Expr *root;
	EpochValue *epoch;
  public:
	EpochExpr(Expr *root, EpochValue *epoch);
	const char *GetPrintNameForNode() { return "EpochExpr"; }
    	void PrintChildren(int indentLevel);	    	
};

class FieldAccess : public Expr {
  protected:
	Expr *base;
	Identifier *field;
  public:
	FieldAccess(Expr *base, Identifier *field, yyltype loc);	
	const char *GetPrintNameForNode() { return "FieldAccess"; }
    	void PrintChildren(int indentLevel);	    	
};

class RangeExpr : public Expr {
  protected:
	Identifier *index;
	Expr *range;
	Expr *step; 
  public:
	RangeExpr(Identifier *index, Expr *range, Expr *step, yyltype loc);		
	const char *GetPrintNameForNode() { return "RangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
};

class SubpartitionRangeExpr : public Expr {
  protected:
	char spaceId;
  public:
	SubpartitionRangeExpr(char spaceId, yyltype loc);
	const char *GetPrintNameForNode() { return "SubpartitionRangeExpr"; }
    	void PrintChildren(int indentLevel);	    	
};

class AssignmentExpr : public Expr {
  protected:
	Expr *left;
	Expr *right;
  public:
	AssignmentExpr(Expr *left, Expr *right, yyltype loc);	
	const char *GetPrintNameForNode() { return "AssignmentExpr"; }
    	void PrintChildren(int indentLevel);	    	
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "ArrayAccess"; }
    	void PrintChildren(int indentLevel);	    	
};

class FunctionCall : public Expr {
  protected:
	Identifier *base;
	List<Expr*> *arguments;
  public:
	FunctionCall(Identifier *base, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "FunctionCall"; }
    	void PrintChildren(int indentLevel);	    	
};

class OptionalInvocationParams : public Expr {
  protected:
	Identifier *section;
	List<Expr*> *arguments;
  public:
	static const char *InitializeSection, *PartitionSection;
	OptionalInvocationParams(Identifier *section, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "OptionalParameters"; }
    	void PrintChildren(int indentLevel);	    	
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
};

class ObjectCreate : public Expr {
  protected:
	Type *objectType;
	List<Expr*> *initArgs;
  public:
	ObjectCreate(Type *objectType, List<Expr*> *initArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "ObjectCreate"; }
    	void PrintChildren(int indentLevel);	    	
};

#endif
