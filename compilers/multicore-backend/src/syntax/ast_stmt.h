#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "../utils/list.h"
#include "ast.h"
#include "../semantics/scope.h"
#include "../static-analysis/data_access.h"
#include "../utils/hashtable.h"

#include <sstream>

class Expr;
class LogicalExpr;
class ReductionExpr;
class Space;
class IndexScope;
class IncludesAndLinksMap;	

class Stmt : public Node {
  public:
	Stmt() : Node() {}
     	Stmt(yyltype loc) : Node(loc) {}

	// Semantic checking is done by recursively going through each statement within a code block 
	// and then by examining each expression within that statement. All sub-classes of statement 
	// therefore should provide an implementation for this method. The symantic analysis process 
	// should have three steps. First, we do the analysis without concern for missing types. This 
	// step should reveal the types of some undeclared variables and some expression. Second, we 
	// do the type inference (using the subsequent method) to recover the missing types. Finally,
	// we do the analysis again with strict type confirmance. If some type information  is still 
	// missing in the final step, we throw type errors. 	 
	virtual void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures) {}
	
	// Since we require the types of only task-global variables to be declared, a type inference 
	// step is needed before we can proceed to semantic analysis.
	virtual void performTypeInference(Scope *executionScope) {}

	// The first stage of static analysis is to determine which task global variable has been 
	// modified where. In that regard, statement class just facilitate a recursive analysis by 
	// invoking the access checking method in all nested expressions. The Expr class does the 
	// actual heavy lifting.
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(
				TaskGlobalReferences *globalReferences) {
		return new Hashtable<VariableAccess*>;
	}
	static void mergeAccessedVariables(Hashtable<VariableAccess*> *first, 
				Hashtable<VariableAccess*> *second);

	// this function is used to recursively determine all the header file includes and library
        // links for different extern code blocks present in an IT-task
        virtual void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {}

	// a back end code generation routine; subclasses should provide appropriate implementations
	virtual void generateCode(std::ostringstream &stream, 
			int indentLevel, Space *space = NULL) {};
};

class StmtBlock : public Stmt {
  protected:
    	List<Stmt*> *stmts;
  public:
    	StmtBlock(List<Stmt*> *statements);
    
	//-------------------------------------------------------------------------Syntax Analysis Routines	
    	const char *GetPrintNameForNode() { return "StmtBlock"; }
    	void PrintChildren(int indentLevel);
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
	List<Stmt*> *getStmtList() { return stmts; }
    
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
	
	//-------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class ConditionalStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *stmt;
  public:
	ConditionalStmt(Expr *condition, Stmt *stmt, yyltype loc);	
    
	//-------------------------------------------------------------------------Syntax Analysis Routines	
    	const char *GetPrintNameForNode() { return "ConditionalStmt"; }
    	void PrintChildren(int indentLevel);
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
	
	//-------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, bool first, Space *space);
};

class IfStmt: public Stmt {
  protected:
	List<ConditionalStmt*> *ifBlocks;
  public:
	IfStmt(List<ConditionalStmt*> *ifBlocks, yyltype loc);	
    
	//-------------------------------------------------------------------------Syntax Analysis Routines	
    	const char *GetPrintNameForNode() { return "IfBlock"; }
    	void PrintChildren(int indentLevel);
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(
			TaskGlobalReferences *globalReferences);
	void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
	
	//-------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
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
    
	//-------------------------------------------------------------------------Syntax Analysis Routines	
    	const char *GetPrintNameForNode() { return "RangeCondition"; }
    	void PrintChildren(int indentLevel);
	List<Identifier*> *getIndexes() { return indexes; }
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void resolveTypes(Scope *executionScope, bool ignoreTypeFailures);
	void inferTypes(Scope *executionScope);
	void putIndexesInIndexScope();
	void validateIndexAssociations(Scope *scope, bool ignoreFailures);
	
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(
			TaskGlobalReferences *globalReferences);

	//-------------------------------------------------------------------------Code Generation Routines
	LogicalExpr *getRestrictions();
};

class LoopStmt: public Stmt {
  protected:
	Scope *scope;
	IndexScope *indexScope;
	Stmt *body;

	// when a loop embodies a reduction operation, it needs to be translated in a way different
	// than normal loop translation. Furthermore, currently the compiler cannot handle other
	// statements within the same loop containing a reduction. To serve both of these needs, we
	// need a flag separating reduction loops from other loops.
	bool reductionLoop;
	// need to maintain a reference to the reduction expression when this embodies a reduction
	ReductionExpr *reduction;
  public:
	LoopStmt();
     	LoopStmt(yyltype loc);
	
	// These two pointers are used to determine what is the closest enclosing loop for a 
	// reduction operation (possibly within the loop). During semantic analysis of the code, as 
	// a loop is encountered and exited, these two references are updated accordingly to be 
	// accessed by any embedded reduction operation.   
	static LoopStmt *currentLoop;
	LoopStmt *previousLoop;
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void setScope(Scope *scope) { this->scope = scope; }
	Scope *getScope() { return scope; }
	void setIndexScope(IndexScope *indexScope);
	IndexScope *getIndexScope();
	void flagAsReductionLoop() { reductionLoop = true; }
	bool isReductionLoop() { return reductionLoop; }
	void setReductionExpr(ReductionExpr *reduction);
	virtual void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);

	//-------------------------------------------------------------------------Static Analysis Routines
	virtual List<const char*> *getIndexNames() = 0;
	void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
	
	//-------------------------------------------------------------------------Code Generation Routines
	// a helper routine for code generation that declares variables in the scope
	void declareVariablesInScope(std::ostringstream &stream, int indentLevel);	
	void generateIndexLoops(std::ostringstream &stream, int indentLevel, 
			Space *space, Stmt *body, List<LogicalExpr*> *indexRestrictions = NULL);
	// a helper routine that decides what index restrictions can be applied to the current 
	// index loop iteration among the list of such restrictions and creates and filters in 
	// the remaining expressions in an argument list
	// @param indexesInvisible is a hashmap of indexes that will be available in some lower
	//        level nested loop; there not visible to the current loop
	// @param currentExprList is the list of restrictions that are still not been applied to
	//        some outer level for loop and therefore subject to consideration
	// @param remainingExprList the list that should hold the expressions that do not enter 
	//        the current logical expression
	// @return a list of logical expressions that are applicable to current Loop  
	List<LogicalExpr*> *getApplicableExprs(Hashtable<const char*> *indexesInvisible, 
			List<LogicalExpr*> *currentExprList, 
			List<LogicalExpr*> *remainingExprList);
	// a routine to declare any supplementary variable and do initialization of the result 
	// variable when the loop is a reduction operation  
	void initializeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space);
	// counterpart routine to terminate/finalize the result of reduction after iteration ends
	void finalizeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space);
	// a routine to carry on actual reduction operation inside the loop
	void performReduction(std::ostringstream &stream, int indentLevel, Space *space);
};

class PLoopStmt: public LoopStmt {
  protected:
	List<IndexRangeCondition*> *rangeConditions;
  public:
	PLoopStmt(List<IndexRangeCondition*> *rangeConditions, Stmt *body, yyltype loc);	
    
	//--------------------------------------------------------------------------Syntax Analysis Routines	
    	const char *GetPrintNameForNode() { return "ParallelForLoop"; }
    	void PrintChildren(int indentLevel);
    
	//------------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	//--------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<const char*> *getIndexNames();
	
	//--------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
	// a function for retrieving a list of boolean expressions that may be part of the range 
	// conditions associated with this loop. A mechanism was needed to break such additional 
	// index traversal restrictions into simpler boolean expressions and then place those 
	// expressions in appropriate for loops to avoid unnecessary iterations in nested loops.
	List<LogicalExpr*> *getIndexRestrictions();
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
};

class SLoopStmt: public LoopStmt {
  protected:
	Identifier *id;
	Expr *rangeExpr;
	Expr *stepExpr;
	Expr *restriction;

	// a flag used for code generation efficiency; if the sequential loop is traversing some
	// array dimension then its code can be generated by invoking the utility method in loop
	// statement class 
	bool isArrayIndexTraversal;
  public:
	SLoopStmt(Identifier *id, SLoopAttribute *attr, Stmt *body, yyltype loc);	
    
	//-------------------------------------------------------------------------Syntex Analysis Routines	
    	const char *GetPrintNameForNode() { return "SequentialForLoop"; }
    	void PrintChildren(int indentLevel);
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	List<const char*> *getIndexNames();
	
	//-------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

class WhileStmt: public Stmt {
  protected:
	Expr *condition;
	Stmt *body;
  public:
	WhileStmt(Expr *condition, Stmt *body, yyltype loc);	
    
	//-------------------------------------------------------------------------Syntex Analysis Routines	
	const char *GetPrintNameForNode() { return "WhileLoop"; }
    	void PrintChildren(int indentLevel);
    
	//-----------------------------------------------------------------------Semantic Analysis Routines	
	void performTypeInference(Scope *executionScope);
	void checkSemantics(Scope *excutionScope, bool ignoreTypeFailures);
    
	//-------------------------------------------------------------------------Static Analysis Routines	
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
	
	//-------------------------------------------------------------------------Code Generation Routines
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
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

        //-------------------------------------------------------------------------Syntex Analysis Routines     
        const char *GetPrintNameForNode() { return "ExternalCodeBlock"; }
        void PrintChildren(int indentLevel);

        //-------------------------------------------------------------------------Static Analysis Routines     
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);

        //-------------------------------------------------------------------------Code Generation Routines
        void generateCode(std::ostringstream &stream, int indentLevel, Space *space);
};

#endif

