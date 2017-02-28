#include "symbol.h"
#include "scope.h"
#include "../syntax/ast.h"
#include "../syntax/ast_type.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include "iostream"
#include "string"

//------------------------------------------------------------- Symbol -----------------------------------------------------------/

Symbol::Symbol(const char *name, Node *node) {
        this->name = name;
        this->node = node;
        this->nestedScope = NULL;
}

void Symbol::describe(int indent) {
	for(int i = 0; i < indent; i++) {
		printf("\t");
	}
}

//--------------------------------------------------------- Variabel Symbol -----------------------------------------------------/

VariableSymbol::VariableSymbol(VariableDef *def) : Symbol(def->getId()->getName(), def) {
	id = def->getId();
	type = def->getType();
	reduction = false;
}

VariableSymbol::VariableSymbol(Identifier *id, Type *type, Node *node) : Symbol(id->getName(), node) {
	this->id = id;
	this->type = type;
	reduction = false;
}

void VariableSymbol::describe(int indent) {
	Symbol::describe(indent);
	printf("Variable %s: %s\n", name, type->getName());
}

//---------------------------------------------------------- Tuple Symbol ------------------------------------------------------/

TupleSymbol::TupleSymbol(Identifier *id, TupleDef *def, List<Type*> *elementTypes) 
		: Symbol(id->getName(), def) {
	this->elementTypes = elementTypes;
}

void TupleSymbol::describe(int indent) {
	Symbol::describe(indent);
	printf("Tuple %s:\n", name);
}

TupleDef *TupleSymbol::getTupleDef() { return (TupleDef*) node; }

//-------------------------------------------------------- Function Symbol -----------------------------------------------------/

FunctionSymbol::FunctionSymbol(FunctionDef *def, TupleDef *returnType, List<Type*> *argumentTypes) 
		: Symbol(def->getId()->getName(), def) {
	this->returnType = returnType;
	this->argumentTypes = argumentTypes;
}

void FunctionSymbol::describe(int indent) {
	Symbol::describe(indent);
	printf("Function %s:\n", name);
}

//---------------------------------------------------------- Task Symbol -------------------------------------------------------/

void TaskSymbol::describe(int indent) {
	Symbol::describe(indent);
	printf("Task %s:\n", name);
	Symbol::describe(indent);
	printf("\tInit Scope:\n");
	initScope->describe(indent + 2);
}
