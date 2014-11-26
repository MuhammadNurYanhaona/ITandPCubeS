#include "hashtable.h"
#include "scope.h"
#include "symbol.h"
#include "iostream"
#include "string"

Scope::Scope(ScopeType type) {
        this->type = type;
        this->parent = NULL;
        symbolTable = new Hashtable<Symbol*>;
	if (type == ComputationStageScope
            		|| type == TaskInitBodyScope
                        || type == FunctionBodyScope
                        || type == CoordinatorScope) {
		supportTypeInference = true;
	} else {
		supportTypeInference = false;
	}
}

Scope* Scope::enter_scope(Scope *newScope) {
        newScope->parent = this;
        return newScope;
}

Scope* Scope::exit_scope() {
        Scope* oldScope = this->parent;
        this->parent = NULL;
        return oldScope;
}

void Scope::insert_symbol(Symbol *symbol) {
        this->symbolTable->Enter(symbol->getName(), symbol, false);
        symbol->setScopeType(this->type);
}

void Scope::copy_symbol(Symbol *symbol) {
        this->symbolTable->Enter(symbol->getName(), symbol, false);
}

Symbol* Scope::lookup(const char *key) {
        Symbol *symbol = this->local_lookup(key);
        if (symbol != NULL) return symbol;
        if (this->parent == NULL) return NULL;
        else return this->parent->lookup(key);
}

Symbol* Scope::local_lookup(const char *key) {
        Symbol *symbol = symbolTable->Lookup(key);
        return symbol;
}

void Scope::detach_from_parent() {
	parent = NULL;
}

bool Scope::insert_inferred_symbol(Symbol *symbol) {
	Scope *scope = this;
	while (scope != NULL && scope->supportTypeInference != true) {
		scope = scope->parent;
	}
	if (scope != NULL) {
		scope->insert_symbol(symbol);
		return true;
	}
	return false;
}

void Scope::describe(int indent) {
        Iterator<Symbol*> symbols = symbolTable->GetIterator();
        Symbol *symbol;
        while((symbol = symbols.GetNextValue()) != NULL) {
                symbol->describe(indent);
		if (symbol->getNestedScope() != NULL) {
                        symbol->getNestedScope()->describe(indent + 1);
                }
        }
}


