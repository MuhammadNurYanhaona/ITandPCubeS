#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "ast_library_fn.h"
#include "list.h"
#include "scope.h"
#include "errors.h"

//-------------------------------------------------------- Static Constants -----------------------------------------------------/

const char *Root::Name = "root";
const char *Random::Name = "random";
const char *LoadArray::Name = "load_array";
const char *LoadListOfArrays::Name = "load_list_of_arrays";
const char *StoreArray::Name = "store_array";
const char *StoreListOfArrays::Name = "store_list_of_arrays";

//-------------------------------------------------------- Library Function -----------------------------------------------------/

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, 
		List<Expr*> *arguments, yyltype loc) : Expr(loc) {
	this->argumentCount = argumentCount;
	this->functionName = functionName;
	this->arguments = arguments;
}

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments) : Expr() {
	this->argumentCount = argumentCount;
	this->functionName = functionName;
	this->arguments = arguments;
}

bool LibraryFunction::isLibraryFunction(Identifier *id) {
	const char* name = id->getName();
	return (strcmp(name, Root::Name) == 0 || strcmp(name, Random::Name) == 0
		|| strcmp(name, LoadArray::Name) == 0 || strcmp(name, LoadListOfArrays::Name) == 0
		|| strcmp(name, StoreArray::Name) == 0 
		|| strcmp(name, StoreListOfArrays::Name) == 0);
}

void LibraryFunction::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}

void LibraryFunction::resolveType(Scope *scope, bool ignoreFailure) {
	if (argumentCount != arguments->NumElements()) {
		ReportError::TooFewOrTooManyParameters(functionName, arguments->NumElements(),
                		argumentCount, ignoreFailure);
	} else {
		validateArguments(scope, ignoreFailure);
	}
}

LibraryFunction *LibraryFunction::getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc) {
	
	const char* name = id->getName();
	LibraryFunction *function = NULL;

	if (strcmp(name, Root::Name) == 0) {
		function = new Root(id, arguments, loc);
	} else if (strcmp(name, Random::Name) == 0) {
		function = new Random(id, arguments, loc);
	} else if (strcmp(name, LoadArray::Name) == 0) {
		function = new LoadArray(id, arguments, loc);
	} else if (strcmp(name, LoadListOfArrays::Name) == 0) {
		function = new LoadListOfArrays(id, arguments, loc);
	} else if (strcmp(name, StoreArray::Name) == 0) {
		function = new StoreArray(id, arguments, loc);
	} else if (strcmp(name, StoreListOfArrays::Name) == 0) {
		function = new StoreListOfArrays(id, arguments, loc);
	}

	return function;	
}

//------------------------------------------------------------ Root ------------------------------------------------------------/

void Root::validateArguments(Scope *scope, bool ignoreFailure) {

	Expr *arg1 = arguments->Nth(0);
	Expr *arg2 = arguments->Nth(1);
	arg1->resolveType(scope, ignoreFailure);
	arg2->resolveType(scope, ignoreFailure);

	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		//TODO report error
	} else if (arg1Type != Type::intType && arg1Type != Type::floatType 
			&& arg1Type != Type::doubleType && arg1Type != Type::errorType) {
		//TODO report error
	}

	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		//TODO report error
	} else if (arg2Type != Type::intType && arg2Type != Type::errorType) {
		//TODO report error
	}
	
	this->type = arg1Type;
}

void Root::inferType(Scope *scope, Type *rootType) {
	if (this->type == NULL) {
		this->type = rootType;
	}
	if (arguments->NumElements() == 2) {
		arguments->Nth(0)->inferType(scope, this->type);
		arguments->Nth(1)->inferType(scope, Type::intType);
	}
}

//------------------------------------------------------- Load Array ---------------------------------------------------------/

void LoadArray::validateArguments(Scope *scope, bool ignoreFailure) {
	Expr *arg1 = arguments->Nth(0);
	arg1->resolveType(scope, ignoreFailure);
	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		//TODO report error	
	} else if (arg1Type != Type::stringType && arg1Type != Type::errorType) {
		//TODO report error
	}	
}

void LoadArray::inferType(Scope *scope, Type *rootType) {
	if (rootType == NULL) {
		this->type = Type::errorType;
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(rootType);
		if (arrayType == NULL) {
			this->type = Type::errorType;
		} else {
			this->type = arrayType;
		}
	}
	if (arguments->NumElements() == 1) {
		arguments->Nth(0)->inferType(scope, Type::stringType);
		arguments->Nth(0)->resolveType(scope, false);
	}
}

//------------------------------------------------------- Store Array ------------------------------------------------------/

void StoreArray::validateArguments(Scope *scope, bool ignoreFailure) {

	Expr *arg1 = arguments->Nth(0);
	arg1->resolveType(scope, ignoreFailure);
	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		//TODO report error	
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(arg1Type);
		if (arrayType == NULL) {
			//TODO report error
		}
	}
	
	Expr *arg2 = arguments->Nth(1);
	arg2->resolveType(scope, ignoreFailure);
	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		//TODO report error	
	} else if (arg2Type != Type::stringType && arg2Type != Type::errorType) {
		//TODO report error
	}
}

void StoreArray::inferType(Scope *scope, Type *rootType) {
	if (rootType == NULL) {
		this->type = Type::errorType;
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(rootType);
		if (arrayType == NULL) {
			this->type = Type::errorType;
		} else {
			this->type = arrayType;
		}
	}
	if (arguments->NumElements() == 2) {
		arguments->Nth(0)->inferType(scope, this->type);
		arguments->Nth(0)->resolveType(scope, false);
		arguments->Nth(1)->inferType(scope, Type::stringType);
		arguments->Nth(1)->resolveType(scope, false);
	}
}
