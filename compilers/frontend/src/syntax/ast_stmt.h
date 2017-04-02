#ifndef _H_ast_stmt
#define _H_ast_stmt

#include "ast.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

class Expr;
class ReductionVar;
class FieldAccess;
class Scope;
class Space;
class PartitionHierarchy;
class Type;
class ParamReplacementConfig;
class TaskGlobalReferences;
class VariableAccess;
class ReductionMetadata;
class IncludesAndLinksMap;

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
	// expressions type-less or having error type. The return types indicates the number of errors.
	virtual int emitScopeAndTypeErrors(Scope *scope) = 0;

	// Subclasses should implement this function to support the polymorphic compute stage resolultion
	// process. Note that we do not intend to produce copies of arrays (or parts of arrays) to generate 
	// stage parameters to realize the stage invocations as templated function calls in the back-ends. 
	// Copying array is costly. Furthermore, templated functions does not fit well into the implicit 
	// typing mechanism of IT. So our scheme is to replace the use of parameters in stage definitions 
	// with that of arguments. This mechanism requires either changing the names of referred fields and 
	// arrays in statements and expressions; or for array parts, in particular, updating the array access 
	// expressions. The two map arguments of this function contains all instructions for parameter 
	// replacement.
	virtual void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) = 0;

	// One aspect of semantic analysis is to determine which task global variable been modified where. 
	// This is needed to ensure that access to variables from specific LPSes are valid. In that regard, 
	// statement class just facilitate a recursive analysis by invoking the access checking method in all 
	// nested expressions. The Expr class does the actual heavy lifting.
        virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs) = 0;
	// A static utility routine added to help data access analysis done using the above interface method 
        static void mergeAccessedVariables(Hashtable<VariableAccess*> *first, 
			Hashtable<VariableAccess*> *second);
        
	//-------------------------------------------------------------------- Helper functions for Static Analysis

	// this interface is used to set up the proper epoch versions to all variables used in expressions 
	// that are parts of a statement 
        virtual void analyseEpochDependencies(Space *space) = 0;

	// this function is used for discovering all reduction statements within a compute stage
        virtual void extractReductionInfo(List<ReductionMetadata*> *infoSet,
                        PartitionHierarchy *lpsHierarchy,
                        Space *executingLps) {}

	//------------------------------------------------------------- Common helper functions for Code Generation
	
	// this function is used to recursively determine all the header file includes and library links 
	// for different extern code blocks present in an IT task
        virtual void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap) {}
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
	void extractReductionInfo(List<ReductionMetadata*> *infoSet,
                        PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
	void extractReductionInfo(List<ReductionMetadata*> *infoSet,
                        PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
	void extractReductionInfo(List<ReductionMetadata*> *infoSet,
                        PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
};

class LoopStmt: public Stmt {
  protected:
	Stmt *body;
	
	// this scope is needed to declare the index variables that are used to traverse index ranges
	Scope *scope;
  public:
	LoopStmt();
     	LoopStmt(Stmt *body, yyltype loc);
	virtual void extractReductionInfo(List<ReductionMetadata*> *infoSet,
			PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
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
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
	void extractReductionInfo(List<ReductionMetadata*> *infoSet,
			PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
};

class ReductionStmt: public Stmt {
  protected:
        Identifier *left;
        ReductionOperator op;
        Expr *right;

	// Having a non-NULL reduction variable associated means this reduction statement's result should
	// be shared among all the LPUs descending from the same ancestor LPU in the LPS indicated by the 
	// reduction variable. If this is NULL then the reduction statement should evaluate locally in each 
	// LPU. 
	ReductionVar *reductionVar;
  public:
        ReductionStmt(Identifier *left, char *opName, Expr *right, yyltype loc);
        const char *GetPrintNameForNode() { return "Reduction-Statement"; }
        void PrintChildren(int indentLevel);

        //------------------------------------------------------------------ Helper functions for Semantic Analysis

	ReductionStmt(Identifier *l, ReductionOperator o, Expr *r, yyltype loc);
        Node *clone();
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration);
	int emitScopeAndTypeErrors(Scope *scope);
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences);

  protected:
	// The reduction operator can be used not only for inferring the type of the expression being reduced
	// but also sometimes for inferring the type type of the result variable. This function embodies the
	// logic of inferring a result variable type given the reduced expression type as an argument.
	Type *inferResultTypeFromOpAndExprType(Type *exprType);

  public:	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
	void extractReductionInfo(List<ReductionMetadata*> *infoSet,
			PartitionHierarchy *lpsHierarchy, 
			Space *executingLps);
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

	// there is no meaningful implementation for any of these functions as an external code block is
	// taken as a whole and applied in the generated code without any analysis from the IT compiler
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId) {}
	int resolveExprTypesAndScopes(Scope *executionScope, int iteration) { return 0; }
	int emitScopeAndTypeErrors(Scope *scope) { return 0; }
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {}
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
                return new Hashtable<VariableAccess*>;
	}
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

	// there is no meaningful implementation for any of these functions as an external code block is
	// taken as a whole and applied in the generated code without any analysis from the IT compiler
        void analyseEpochDependencies(Space *space) {}
	
	//------------------------------------------------------------- Common helper functions for Code Generation
	
        void retrieveExternHeaderAndLibraries(IncludesAndLinksMap *includesAndLinksMap);
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
	int emitScopeAndTypeErrors(Scope *scope);

	// there is no action in this regard on return statements as compute stages cannot have return 
	// statements -- only IT functions can have them
	void performStageParamReplacement(
			Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
			Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap) {}
	
	// there is no action for this also as a return statement is only meaningful inside a function which
	// cannot directly access any task global variable
	Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
                return new Hashtable<VariableAccess*>;
	}
	
	//-------------------------------------------------------------------- Helper functions for Static Analysis

        void analyseEpochDependencies(Space *space);
};

#endif

