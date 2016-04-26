#include "../utils/hashtable.h"
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

void Scope::remove_symbol(const char *key) {
	Symbol *symbol = local_lookup(key);
	if (symbol != NULL) {
		this->symbolTable->Remove(key, symbol);
	}
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

Scope *Scope::get_nearest_scope(ScopeType type) {
	if (this->type == type) return this;
	if (this->parent == NULL) return NULL;
	return parent->get_nearest_scope(type);
}

void Scope::declareVariables(std::ostringstream &stream, int indent) {

        std::string stmtSeparator = ";\n";
        std::ostringstream stmtIndent;
        for (int i = 0; i < indent; i++) stmtIndent << '\t';

        Iterator<Symbol*> iterator = this->get_local_symbols();
        Symbol *symbol;
        while ((symbol = iterator.GetNextValue()) != NULL) {

                VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                if (variable == NULL) continue;
                Type *type = variable->getType();
                const char *name = variable->getName();
                stream << stmtIndent.str() << type->getCppDeclaration(name) << stmtSeparator;

                // if the variable is a dynamic array then we need to a declare metadata variable
                // alongside its own declaration
                ArrayType *array = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                if (array != NULL && staticArray == NULL) {
                        int dimensions = array->getDimensions();
                        stream << stmtIndent.str() << "PartDimension " << name << "Dims";
                        stream << "[" << dimensions << "]" << stmtSeparator;
			stream << stmtIndent.str() << "ArrayTransferConfig " << name;
			stream << "TransferConfig = ArrayTransferConfig()" << stmtSeparator;
                }
        }
}

