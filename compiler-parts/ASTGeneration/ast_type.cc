/* File: ast_type.cc
 * -----------------
 * Implementation of type node classes.
 */
#include "ast_type.h"
#include <string.h>

/* Class constants
 * ---------------
 * These are public constants for the built-in base types (int, double, etc.)
 * They can be accessed with the syntax Type::intType. This allows you to
 * directly access them and share the built-in types where needed rather that
 * creates lots of copies.
 */

Type *Type::intType    		= new Type("int");
Type *Type::floatType  		= new Type("float");
Type *Type::doubleType 		= new Type("double");
Type *Type::charType   		= new Type("char");
Type *Type::boolType   		= new Type("bool");
Type *Type::stringType 		= new Type("string");
Type *Type::epochType   	= new Type("epoch");
Type *Type::dimensionType  	= new Type("dimension");
Type *Type::rangeType   	= new Type("range");
Type *Type::indexType   	= new Type("index");
Type *Type::errorType  		= new Type("error"); 

Type::Type(const char *n) {
    	Assert(n);
    	typeName = strdup(n);
}

void Type::PrintChildren(int indentLevel) {
    	printf("%s", typeName);
}

	
NamedType::NamedType(Identifier *i) : Type(*i->GetLocation()) {
    	Assert(i != NULL);
    	(id=i)->SetParent(this);
} 

void NamedType::PrintChildren(int indentLevel) {
    	id->Print(indentLevel+1);
}

ArrayType::ArrayType(yyltype loc, Type *et, int dims) : Type(loc) {
    	Assert(et != NULL && dims != 0);
  	dimensions = dims;	
    	(elemType=et)->SetParent(this);
}
void ArrayType::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Dimensions");
	printf("%d", dimensions);
    	elemType->Print(indentLevel + 1, "Element ");
}

void StaticArrayType::setLengths(List<int> *dimLengths) {
	dimensionLengths = dimLengths;
}

void StaticArrayType::PrintChildren(int indentLevel) {
	ArrayType::PrintChildren(indentLevel);
	PrintLabel(indentLevel + 1, "Dimension Lengths");
	for (int i = 0; i < dimensionLengths->NumElements(); i++) {
		printf("%d ", dimensionLengths->Nth(i));
	}
}

ListType::ListType(yyltype loc, Type *et) : Type(loc) {
    	Assert(et != NULL);
    	(elemType=et)->SetParent(this);
}

void ListType::PrintChildren(int indentLevel) {
    	elemType->Print(indentLevel + 1, "Element ");
}


