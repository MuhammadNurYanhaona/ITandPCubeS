/* File: ast_stmt.h
 * ----------------
 * The Stmt class and its subclasses are used to represent
 * statements in the parse tree.  For each statment in the
 * language (for, if, return, etc.) there is a corresponding
 * node class for that construct. 
 */

#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "../utils/list.h"
#include "ast.h"
#include "../semantics/scope.h"
#include "../static-analysis/data_access.h"
#include "../utils/hashtable.h"

#include <sstream>

class Expr;
class Space;

class Stmt : public Node {
  public:
	Stmt() : Node() {}
     	Stmt(yyltype loc) : Node(loc) {}

	// Semantic checking is done by recursively going through each statement within
	// a code block and then by examining each expression within that statement. All
	// sub-classes of statement therefore should provide an implementation for this
	// method. The symantic analysis process should have three steps. First, we do 
	// the analysis without concern for missing types. This step should reveal the
	// types of some undeclared variables and some expression. Second, we do the type 
	// inference (using the subsequent method) to recover the missing types. Finally,
	// we do the analysis again with strict type confirmance. If some type information
	// is still missing in the final step, we throw type errors. 	 
	virtual void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures) {}
	
	// Since we require the types of only task-global variables to be declared, a type
	// inference step is needed before we can proceed to semantic analysis.
	virtual void performTypeInference(Scope *executionScope) {}

	// The first stage of static analysis is to determine which task global variable
	// been modified where. In that regard, statement class just facilitate a recursive
	// analysis by invoking the access checking method in all nested expressions. The
	// Expr class does the actual heavy lifting.
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(
				TaskGlobalReferences *globalReferences) {
		return new Hashtable<VariableAccess*>;
	}
	static void mergeAccessedVariables(Hashtable<VariableAccess*> *first, 
				Hashtable<VariableAccess*> *second);

	// back end code generation routine; subclasses should provide appropriate 
	// implementations
	virtual void generateCode(std::ostringstream &stream, int indentLevel, Space *space) {};
};

class StmtBlock : public Stmt {
  protected:
    	List<Stmt*> *stmts;

  public:
    	StmtBlock(List<Stmt*> *statements);
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "StmtBlock"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class ConditionalStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *stmt;
  public:
	ConditionalStmt(Expr *condition, Stmt *stmt, yyltype loc);	
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "ConditionalStmt"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, bool first, Space *space);
};

class IfStmt: public Stmt {
  protected:
	List<ConditionalStmt*> *ifBlocks;
  public:
	IfStmt(List<ConditionalStmt*> *ifBlocks, yyltype loc);	
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "IfBlock"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class IndexRangeCondition: public Node {
  protected:
	List<Identifier*> *indexes;
	Identifier *collection;
	Expr *restrictions;
  public:
	IndexRangeCondition(List<Identifier*> *indexes, 
			Identifier *collection, Expr *restrictions, yyltype loc);		
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "RangeCondition"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void resolveTypes(Scope *executionScope, bool ignoreTypeFailures);
	void inferTypes(Scope *executionScope);
	void putIndexesInIndexScope();
	void validateIndexAssociations(Scope *scope, bool ignoreFailures);
	
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
};

class LoopStmt: public Stmt {
  protected:
	Scope *scope;
  public:
	LoopStmt() : Stmt() { scope = NULL; }
     	LoopStmt(yyltype loc) : Stmt(loc) { scope == NULL; }
    
	// Semantic Analysis Routines	
	void setScope(Scope *scope) { this->scope = scope; }
	Scope *getScope() { return scope; }
	
	// a helper routine for code generation that declares variables in the scope
	void declareVariablesInScope(std::ostringstream &stream, int indentLevel);	
};

class PLoopStmt: public LoopStmt {
  protected:
	List<IndexRangeCondition*> *rangeConditions;
	Stmt *body;
  public:
	PLoopStmt(List<IndexRangeCondition*> *rangeConditions, Stmt *body, yyltype loc);	
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "ParallelForLoop"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
};

class SLoopStmt: public LoopStmt {
  protected:
	Identifier *id;
	Expr *rangeExpr;
	Expr *stepExpr;
	Stmt *body;
  public:
	SLoopStmt(Identifier *id, Expr *rangeExpr, Expr *stepExpr, Stmt *body, yyltype loc);	
    
	// Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "SequentialForLoop"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class WhileStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *body;
  public:
	WhileStmt(Expr *condition, Stmt *body, yyltype loc);	
    
	// Syntex Analysis Routines	
	const char *GetPrintNameForNode() { return "WhileLoop"; }
    	void PrintChildren(int indentLevel);
    
	// Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	// Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	// Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

#endif

