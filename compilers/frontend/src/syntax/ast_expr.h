#ifndef _H_ast_expr
#define _H_ast_expr

#include "ast.h"
#include "ast_stmt.h"
#include "ast_type.h"

#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

class TaskDef;
class FieldAccess;
class Space;

class Expr : public Stmt {
  protected:
	Type *type;
  public:
    	Expr(yyltype loc) : Stmt(loc) { type = NULL; }
    	Expr() : Stmt() { type = NULL; }
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
};

class FloatConstant : public Expr {
  protected:
    	float value;
  public:
    	FloatConstant(yyltype loc, float val);
    	const char *GetPrintNameForNode() { return "Float-Constant"; }
    	void PrintChildren(int indentLevel);
};

class DoubleConstant : public Expr {
  protected:
    	double value;
  public:
    	DoubleConstant(yyltype loc, double val);
    	const char *GetPrintNameForNode() { return "Double-Constant"; }
    	void PrintChildren(int indentLevel);
};

class BoolConstant : public Expr {
  protected:
    	bool value;
  public:
    	BoolConstant(yyltype loc, bool val);
    	const char *GetPrintNameForNode() { return "Boolean-Constant"; }
    	void PrintChildren(int indentLevel);
};

class StringConstant : public Expr {
  protected:
    	const char *value;
  public:
    	StringConstant(yyltype loc, const char *val);
    	const char *GetPrintNameForNode() { return "String-Constant"; }
    	void PrintChildren(int indentLevel);
	const char *getValue() { return value; }
};

class CharConstant : public Expr {
  protected:
    	char value;
  public:
    	CharConstant(yyltype loc, char val);
    	const char *GetPrintNameForNode() { return "Character-Constant"; }
    	void PrintChildren(int indentLevel);
};

class ReductionVar : public Expr {
  protected:
	char spaceId;
	const char *name;
  public:
	ReductionVar(char spaceId, const char *name, yyltype loc);	
    	const char *GetPrintNameForNode() { return "Reduction-Var"; }
    	void PrintChildren(int indentLevel);
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
};

class EpochExpr : public Expr {
  protected:
	Expr *root;
	int lag;
  public:
	EpochExpr(Expr *root, int lag);
	const char *GetPrintNameForNode() { return "Epoch-Expr"; }
    	void PrintChildren(int indentLevel);
};

class FieldAccess : public Expr {
  protected:
	Expr *base;
	Identifier *field;
  public:
	FieldAccess(Expr *base, Identifier *field, yyltype loc);	
	const char *GetPrintNameForNode() { return "Field-Access"; }
    	void PrintChildren(int indentLevel);	    	
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
};

class ArrayAccess : public Expr {
  protected:
	Expr *base;
	Expr *index;
  public:
	ArrayAccess(Expr *base, Expr *index, yyltype loc);		
	const char *GetPrintNameForNode() { return "Array-Access"; }
    	void PrintChildren(int indentLevel);
	Expr *getBase() { return base; }
	Expr *getIndex() { return index; }
};

class FunctionCall : public Expr {
  protected:
	Identifier *base;
	List<Expr*> *arguments;
  public:
	FunctionCall(Identifier *base, List<Expr*> *arguments, yyltype loc);		
	const char *GetPrintNameForNode() { return "Function-Call"; }
    	void PrintChildren(int indentLevel);	    	
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
};

class NamedMultiArgument : public Node {
  protected:
	const char *argName;
	List<Expr*> *argList;
  public:
	NamedMultiArgument(char *argName, List<Expr*> *argList, yyltype loc);
	const char *GetPrintNameForNode() { return "Named-Multi-Argument"; }
    	void PrintChildren(int indentLevel);	    	
};

class TaskInvocation : public Expr {
  protected:
	List<NamedMultiArgument*> *invocationArgs;	
  public:
	TaskInvocation(List<NamedMultiArgument*> *invocationArgs, yyltype loc);
	const char *GetPrintNameForNode() { return "Task-Invocation"; }
    	void PrintChildren(int indentLevel);	    	
};

class ObjectCreate : public Expr {
  protected:
	Type *objectType;
	List<NamedArgument*> *initArgs;
  public:
	ObjectCreate(Type *objectType, List<NamedArgument*> *initArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "Object-Create"; }
    	void PrintChildren(int indentLevel);
};

#endif
