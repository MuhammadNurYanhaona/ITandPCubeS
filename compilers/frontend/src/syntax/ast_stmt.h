#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "ast.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

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
    	const char *GetPrintNameForNode() { return "Statement-Block"; }
    	void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
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
};

class LoopStmt: public Stmt {
  protected:
	Stmt *body;
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
};

#endif

