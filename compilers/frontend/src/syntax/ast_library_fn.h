#ifndef _H_ast_library_fn
#define _H_ast_library_fn

/* This header file contains functions that are available in any IT program as a library support alongside other
   user defined custom functions. These library functions are especially treated during syntax tree generation 
   and can even be used to introduce behavior in an IT code that may be difficult to express in IT source given
   the current syntax and features set of the language. The trick is, one have to write how code generation for
   these functions are done for different back-end environments.
*/

#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <fstream>
#include <sstream>

class Scope;
class VariableAccess;
class TaskGlobalReferences;

class LibraryFunction : public Expr {
  protected:
	List<Expr*> *arguments;
	int argumentCount;
	Identifier *functionName;	
	LibraryFunction(int argumentCount, 
			Identifier *functionName, 
			List<Expr*> *arguments, 
			yyltype loc);
        LibraryFunction(int argumentCount, 
			Identifier *functionName, 
			List<Expr*> *arguments);
  public:
	// any new library functions should be mentioned and initialized in the implementation of these two
	// functions
	static bool isLibraryFunction(Identifier *id);
	static LibraryFunction *getFunctionExpr(Identifier *id, 
			List<Expr*> *arguments, yyltype loc);

	const char *GetPrintNameForNode() { return functionName->getName(); }
        void PrintChildren(int indentLevel);

        //-------------------------------------------------------------- Helper functions for Semantic Analysis

        Node *clone();
	ExprTypeId getExprTypeId() { return LIB_FN_CALL; }
	void retrieveExprByType(List<Expr*> *exprList, ExprTypeId typeId);
	int emitSemanticErrors(Scope *scope);

	// subclasses should provide implementation for this function for semantic validation
	virtual int emitErrorsInArguments(Scope *scope) = 0;

	// function interface inherited from the statement base-class that is used to resolve a type
	// polymorphic compute stage for a particular set of arguments for the parameters 
	virtual void performStageParamReplacement(
                        Hashtable<ParamReplacementConfig*> *nameAdjustmentInstrMap,
                        Hashtable<ParamReplacementConfig*> *arrayAccXformInstrMap);

	// function interface inherited from the statement base-class to be used to validate access of
	// task-global data structures from different LPSes 
	virtual Hashtable<VariableAccess*> *getAccessedGlobalVariables(TaskGlobalReferences *globalRefs);
};


/*------------------------------------------------------------------------------------------------------------- 
	   				Common math functions
-------------------------------------------------------------------------------------------------------------*/

class Root : public LibraryFunction {
  public:
	static const char *Name;	
	Root(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(2, id, arguments, loc) {}

        //-------------------------------------------------------------- Helper functions for Semantic Analysis

	int resolveExprTypes(Scope *scope);
	int inferExprTypes(Scope *scope, Type *assignedType);
	int emitErrorsInArguments(Scope *scope);
};

class Random : public LibraryFunction {
  public:
	static const char *Name;	
	Random(Identifier *id, List<Expr*> *arguments, yyltype loc) 
			: LibraryFunction(0, id, arguments, loc) {
		this->type = Type::intType;
	}
	int emitErrorsInArguments(Scope *scope) { return 0; }
	int countTypeErrors();	
};

/*------------------------------------------------------------------------------------------------------------- 
	       Functions for loading or storing arrays directly in the coordinator program
-------------------------------------------------------------------------------------------------------------*/

class ArrayOperation : public LibraryFunction {
  public:
	ArrayOperation(Identifier *id, List<Expr*> *arguments, yyltype loc)
                : LibraryFunction(2, id, arguments, loc) {}

        //-------------------------------------------------------------- Helper functions for Semantic Analysis

	virtual int resolveExprTypes(Scope *scope);
	virtual int emitErrorsInArguments(Scope *scope);
};

class LoadArray : public ArrayOperation {
  public:
	static const char *Name;	
	LoadArray(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}
};

class StoreArray : public ArrayOperation {
  public:
	static const char *Name;	
	StoreArray(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}	
};

/*------------------------------------------------------------------------------------------------------------- 
	   Functions for instructing a task to get data input or write data output to/from files
-------------------------------------------------------------------------------------------------------------*/

class BindOperation : public LibraryFunction {
  public:
	BindOperation(Identifier *id, List<Expr*> *arguments, yyltype loc)
                : LibraryFunction(3, id, arguments, loc) {}

        //-------------------------------------------------------------- Helper functions for Semantic Analysis

	int resolveExprTypes(Scope *scope);
	virtual int emitErrorsInArguments(Scope *scope);
};

class BindInput : public BindOperation {
  public:
	static const char *Name;
	BindInput(Identifier *id, List<Expr*> *args, yyltype loc) : BindOperation(id, args, loc) {}
};

class BindOutput : public BindOperation {
  public:
	static const char *Name;
	BindOutput(Identifier *id, List<Expr*> *args, yyltype loc) : BindOperation(id, args, loc) {}
};

#endif
