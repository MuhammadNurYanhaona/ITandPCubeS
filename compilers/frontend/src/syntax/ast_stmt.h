#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "ast.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

class Expr;
class Scope;
class Type;

class Stmt : public Node {
  public:
	Stmt() : Node() {}
     	Stmt(yyltype loc) : Node(loc) {}

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

	// This function is needed to filter all nested expressions with a specific type for tagging and
	// further processing.
	virtual void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) = 0;

	// Scope-and-type checking is done by recursively going through each statement within a code block 
	// and then by examining each expression within that statement. All sub-classes of statement class, 
	// therefore should provide an implementation for this method. Since IT has a mixture of implicit
	// and explicit typing, this function has to be invoked again and again to infer the unknown types 
	// of some expressions until a fixed-point has been reached when no new type can be inferred. Whether
	// a particular invocation of this method has resolved any new type is determined by the returned 
	// value. The returned integer indicates the number of new type resolutions. The second argument
	// indicates the round at which the scope-and-type resolution is currently in for special treatment
	// of the first iteration when needed. 
	virtual int resolveExprTypesAndScopes(Scope *executionScope, int iteration) = 0;

	// This function is needed to determine if the type-and-scope resolution process ended with some
	// expressions type-less or having error type.
	virtual int countTypeErrors() = 0;
};

class StmtBlock : public Stmt {
  protected:
    	List<Stmt*> *stmts;
  public:
    	StmtBlock(List<Stmt*> *statements);
    	const char *GetPrintNameForNode() { return "Statement-Block"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class ConditionalStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *stmt;
  public:
	ConditionalStmt(Expr *condition, Stmt *stmt, yyltype loc);	
    	const char *GetPrintNameForNode() { return "Conditional-Stmt"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class IfStmt: public Stmt {
  protected:
	List<ConditionalStmt*> *ifBlocks;
  public:
	IfStmt(List<ConditionalStmt*> *ifBlocks, yyltype loc);	
    	const char *GetPrintNameForNode() { return "If-Block"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class IndexRangeCondition: public Node {
  protected:
	List<Identifier*> *indexes;
	Identifier *collection;
	Expr *restrictions;
	int dimensionNo;
  public:
	IndexRangeCondition(List<Identifier*> *indexes, 
			Identifier *collection, int dimensionNo, 
			Expr *restrictions, yyltype loc);
    	const char *GetPrintNameForNode() { return "Range-Condition"; }
    	void PrintChildren(int indentLevel);
	List<Identifier*> *getIndexes() { return indexes; }

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class LoopStmt: public Stmt {
  protected:
	Stmt *body;
	
	// this scope is needed to declare the index variables that are used to traverse index ranges
	Scope *scope;
  public:
	LoopStmt();
     	LoopStmt(Stmt *body, yyltype loc);
};

class PLoopStmt: public LoopStmt {
  protected:
	List<IndexRangeCondition*> *rangeConditions;
  public:
	PLoopStmt(List<IndexRangeCondition*> *rangeConditions, Stmt *body, yyltype loc);	
    	const char *GetPrintNameForNode() { return "Parallel-For-Loop"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class SLoopAttribute {
  protected:
        Expr *range;
        Expr *step;
        Expr *restriction;
  public:
        SLoopAttribute(Expr *range, Expr *step, Expr *restriction);
        Expr *getRange() { return range; }
        Expr *getStep() { return step; }
        Expr *getRestriction() { return restriction; }

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        SLoopAttribute *clone();
};

class SLoopStmt: public LoopStmt {
  protected:
	Identifier *id;
	Expr *rangeExpr;
	Expr *stepExpr;
	Expr *restriction;
	SLoopAttribute *attrRef;
  public:
	SLoopStmt(Identifier *id, SLoopAttribute *attr, Stmt *body, yyltype loc);	
    	const char *GetPrintNameForNode() { return "Sequential-For-Loop"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class WhileStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *body;
  public:
	WhileStmt(Expr *condition, Stmt *body, yyltype loc);	
	const char *GetPrintNameForNode() { return "While-Loop"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();
};

class ReductionStmt: public Stmt {
  protected:
        Identifier *left;
        ReductionOperator op;
        Expr *right;
  public:
        ReductionStmt(Identifier *left, char *opName, Expr *right, yyltype loc);
        const char *GetPrintNameForNode() { return "Reduction-Statement"; }
        void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

	ReductionStmt(Identifier *l, ReductionOperator o, Expr *r, yyltype loc);
        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int countTypeErrors();

  protected:
	// The reduction operator can be used not only for inferring the type of the expression being reduced
	// but also sometimes for inferring the type type of the result variable. This function embodies the
	// logic of inferring a result variable type given the reduced expression type as an argument.
	Type *inferResultTypeFromOpAndExprType(Type *exprType);
};

class ExternCodeBlock: public Stmt {
  protected:
	const char *language;
	List<const char*> *headerIncludes;
	List<const char*> *libraryLinks;
	const char *codeBlock;
  public:
	ExternCodeBlock(const char *language, 
			List<const char*> *headerIncludes, 
			List<const char*> *libraryLinks, 
			const char *codeBlock, yyltype loc);
	const char *GetPrintNameForNode() { return "External-Code-Block"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {}
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration) { return 0; }
	int countTypeErrors() { return 0; }
};

class ReturnStmt: public Stmt {
  protected:
	Expr *expr;
  public:
	ReturnStmt(Expr *expr, yyltype loc);
	const char *GetPrintNameForNode() { return "Return-Statement"; }
	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();		
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	Expr *getExpr() { return expr; }
	int countTypeErrors();
};

#endif

