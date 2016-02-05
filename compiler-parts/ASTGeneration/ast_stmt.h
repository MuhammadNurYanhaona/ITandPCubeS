/* File: ast_stmt.h
 * ----------------
 * The Stmt class and its subclasses are used to represent
 * statements in the parse tree.  For each statment in the
 * language (for, if, return, etc.) there is a corresponding
 * node class for that construct. 
 */

#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "list.h"
#include "ast.h"

class Expr;

class Stmt : public Node {
  public:
	Stmt() : Node() {}
     	Stmt(yyltype loc) : Node(loc) {}
};

class StmtBlock : public Stmt {
  protected:
    	List<Stmt*> *stmts;

  public:
    	StmtBlock(List<Stmt*> *statements);
    	const char *GetPrintNameForNode() { return "StmtBlock"; }
    	void PrintChildren(int indentLevel);
};

class ConditionalStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *stmt;
  public:
	ConditionalStmt(Expr *condition, Stmt *stmt, yyltype loc);	
    	const char *GetPrintNameForNode() { return "ConditionalStmt"; }
    	void PrintChildren(int indentLevel);
};

class IfStmt: public Stmt {
  protected:
	List<ConditionalStmt*> *ifBlocks;
  public:
	IfStmt(List<ConditionalStmt*> *ifBlocks, yyltype loc);	
    	const char *GetPrintNameForNode() { return "IfBlock"; }
    	void PrintChildren(int indentLevel);
};

class IndexRangeCondition: public Node {
  protected:
	List<Identifier*> *indexes;
	Identifier *collection;
	Expr *restrictions;
  public:
	IndexRangeCondition(List<Identifier*> *indexes, 
			Identifier *collection, Expr *restrictions, yyltype loc);		
    	const char *GetPrintNameForNode() { return "RangeCondition"; }
    	void PrintChildren(int indentLevel);
};

class PLoopStmt: public Stmt {
  protected:
	List<IndexRangeCondition*> *rangeConditions;
	Stmt *body;
  public:
	PLoopStmt(List<IndexRangeCondition*> *rangeConditions, Stmt *body, yyltype loc);	
    	const char *GetPrintNameForNode() { return "ParallelForLoop"; }
    	void PrintChildren(int indentLevel);
};

class SLoopStmt: public Stmt {
  protected:
	Identifier *id;
	Expr *rangeExpr;
	Expr *stepExpr;
	Stmt *body;
  public:
	SLoopStmt(Identifier *id, Expr *rangeExpr, Expr *stepExpr, Stmt *body, yyltype loc);	
    	const char *GetPrintNameForNode() { return "SequentialForLoop"; }
    	void PrintChildren(int indentLevel);
};

class WhileStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *body;
  public:
	WhileStmt(Expr *condition, Stmt *body, yyltype loc);	
    	const char *GetPrintNameForNode() { return "WhileLoop"; }
    	void PrintChildren(int indentLevel);
};

#endif

