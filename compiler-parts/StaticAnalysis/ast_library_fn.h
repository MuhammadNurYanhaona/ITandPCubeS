#ifndef _H_ast_library_fn
#define _H_ast_library_fn

#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "list.h"
#include "scope.h"

class LibraryFunction : public Expr {
  protected:
	List<Expr*> *arguments;
	int argumentCount;
	Identifier *functionName;	
	LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments, yyltype loc);
        LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments);
  public:
	static bool isLibraryFunction(Identifier *id);
	static LibraryFunction *getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc);
	const char *GetPrintNameForNode() { return functionName->getName(); }
        void PrintChildren(int indentLevel);

	void resolveType(Scope *scope, bool ignoreFailure);
	virtual void validateArguments(Scope *scope, bool ignoreFailure) {}
};

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

class LoadArray : public LibraryFunction {
  public:
	static const char *Name;	
	LoadArray(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(1, id, arguments, loc) {}	
	void validateArguments(Scope *scope, bool ignoreFailure);
	virtual void inferType(Scope *scope, Type *type);		
};

class StoreArray : public LibraryFunction {
  public:
	static const char *Name;	
	StoreArray(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(2, id, arguments, loc) {}	
	void validateArguments(Scope *scope, bool ignoreFailure);
	virtual void inferType(Scope *scope, Type *type);		
};

class LoadListOfArrays : public LibraryFunction {
  public:
	static const char *Name;	
	LoadListOfArrays(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(1, id, arguments, loc) {}	
};

class StoreListOfArrays : public LibraryFunction {
  public:
	static const char *Name;	
	StoreListOfArrays(Identifier *id, List<Expr*> *arguments, yyltype loc) 
		: LibraryFunction(2, id, arguments, loc) {}	
};

#endif
