#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "ast_library_fn.h"
#include "../utils/list.h"
#include "../semantics/scope.h"
#include "errors.h"

#include <iostream>

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
		std::cout << "argument count problem\n";
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
		ReportError::UnknownExpressionType(arg1, ignoreFailure);	
	} else if (arg1Type != Type::intType 
			&& arg1Type != Type::floatType 
			&& arg1Type != Type::doubleType 
			&& arg1Type != Type::errorType) {
		ReportError::InvalidExprType(arg1, arg1Type, ignoreFailure);
	}

	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		ReportError::UnknownExpressionType(arg2, ignoreFailure);	
	} else if (arg2Type != Type::intType && arg2Type != Type::errorType) {
		ReportError::IncompatibleTypes(arg2->GetLocation(), arg2Type, Type::intType, ignoreFailure);
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

//--------------------------------------------------- Array Operation -----------------------------------------------------/

void ArrayOperation::validateArguments(Scope *scope, bool ignoreFailure) {
	
	Expr *arg1 = arguments->Nth(0);
	arg1->resolveType(scope, ignoreFailure);
	Type *arg1Type = arg1->getType();
	if (arg1Type == NULL) {
		ReportError::UnknownExpressionType(arg1, ignoreFailure);	
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(arg1Type);
		if (arrayType == NULL) {
			ReportError::InvalidArrayAccess(arg1->GetLocation(), arg1Type, ignoreFailure);
		}
	}

	Expr *arg2 = arguments->Nth(1);
	arg2->resolveType(scope, ignoreFailure);
	Type *arg2Type = arg2->getType();
	if (arg2Type == NULL) {
		ReportError::UnknownExpressionType(arg2, ignoreFailure);	
	} else if (arg2Type != Type::stringType && arg2Type != Type::errorType) {
		ReportError::IncompatibleTypes(arg2->GetLocation(), arg2Type, Type::stringType, ignoreFailure);
	}
	this->type = Type::voidType;	
}

void ArrayOperation::inferType(Scope *scope, Type *rootType) {
	if (arguments->NumElements() == 2) {
		arguments->Nth(1)->inferType(scope, Type::stringType);
		arguments->Nth(1)->resolveType(scope, false);
	}
}

//------------------------------------------------------- Load Array ------------------------------------------------------/
