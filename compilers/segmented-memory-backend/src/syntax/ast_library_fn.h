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
#include "../utils/list.h"
#include "../semantics/scope.h"

#include <fstream>
#include <sstream>

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

	void resolveType(Scope *scope, bool ignoreFailure);
	virtual void validateArguments(Scope *scope, bool ignoreFailure) {}
};


/*------------------------------------------------------------------------------------------------------------- 
	   				Common math functions
-------------------------------------------------------------------------------------------------------------*/

class Root : public LibraryFunction {
  public:
	static const char *Name;	
	Root(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(2, id, arguments, loc) {}	
	virtual void validateArguments(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *type);		
};

class Random : public LibraryFunction {
  public:
	static const char *Name;	
	Random(Identifier *id, List<Expr*> *arguments, yyltype loc) 
			: LibraryFunction(0, id, arguments, loc) {
		this->type = Type::intType;
	}	
};

/*------------------------------------------------------------------------------------------------------------- 
	   Four functions for loading or storing array(s) directly in the coordinator program
-------------------------------------------------------------------------------------------------------------*/

class ArrayOperation : public LibraryFunction {
  public:
	ArrayOperation(Identifier *id, List<Expr*> *arguments, yyltype loc)
                : LibraryFunction(2, id, arguments, loc) {}
	void validateArguments(Scope *scope, bool ignoreFailure);
        virtual void inferType(Scope *scope, Type *type);
};

class LoadArray : public ArrayOperation {
  public:
	static const char *Name;	
	LoadArray(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);	
};

class StoreArray : public ArrayOperation {
  public:
	static const char *Name;	
	StoreArray(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}	
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);	
};

class LoadListOfArrays : public ArrayOperation {
  public:
	static const char *Name;	
	LoadListOfArrays(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}	
};

class StoreListOfArrays : public ArrayOperation {
  public:
	static const char *Name;	
	StoreListOfArrays(Identifier *id, List<Expr*> *arguments, 
			yyltype loc) : ArrayOperation(id, arguments, loc) {}	
};

/*------------------------------------------------------------------------------------------------------------- 
	   Functions for instructing a task to get data input or write data output to/from files
-------------------------------------------------------------------------------------------------------------*/

class BindOperation : public LibraryFunction {
  public:
	BindOperation(Identifier *id, List<Expr*> *arguments, yyltype loc)
                : LibraryFunction(3, id, arguments, loc) {}
	void validateArguments(Scope *scope, bool ignoreFailure);
	void inferType(Scope *scope, Type *type);
};

class BindInput : public BindOperation {
  public:
	static const char *Name;
	BindInput(Identifier *id, List<Expr*> *args, yyltype loc) : BindOperation(id, args, loc) {}
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);	
};

class BindOutput : public BindOperation {
  public:
	static const char *Name;
	BindOutput(Identifier *id, List<Expr*> *args, yyltype loc) : BindOperation(id, args, loc) {}
	void generateCode(std::ostringstream &stream, int indentLevel, Space *space);	
};

#endif
