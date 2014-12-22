#ifndef _H_symbol
#define _H_symbol

#include "../syntax/ast.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast_task.h"
#include "../utils/hashtable.h"
#include "scope.h"

class Symbol {
    protected:
        const char *name;
	Node *node;
	Scope *nestedScope;
        ScopeType scopeType;

    public:
        Symbol(const char *name, Node *node);
        const char* getName() { return name; }
	Node *getNode() { return node; }
        virtual bool isEquivalent(Symbol *otherSymbol) { return this == otherSymbol ; }
	virtual bool isCompatible(Symbol *otherSymbol, Scope *currentContext) { return false; }
        virtual void describe(int indent);
	void setNestedScope(Scope *scope) { this->nestedScope = scope; }
        Scope *getNestedScope() { return nestedScope; }
        void setScopeType(ScopeType scopeType) { this->scopeType = scopeType; }
};

class TaskSymbol : public Symbol {
  protected:
	Scope *initScope;
	Scope *envScope;
	Scope *partitionScope;
  public:
	TaskSymbol(const char *name, TaskDef *def) : Symbol(name, def) {}
	void setInitScope(Scope *scope) { initScope = scope; }
	Scope *getInitScope() { return initScope; }	
	void setEnvScope(Scope *scope) { envScope = scope; }
	Scope *getEnvScope() { return envScope; }	
	void setPartitionScope(Scope *scope) { partitionScope = scope; }
	Scope *getPartitionScope() { return partitionScope; }	
        void describe(int indent);
};

class FunctionSymbol : public Symbol {
  protected:
	TupleDef *returnType;
	List<Type*> *argumentTypes;
  public:
	FunctionSymbol(FunctionDef *def, TupleDef *returnType, List<Type*> *argumentTypes);
	List<Type*> *getArgumentTypes() { return argumentTypes; }
	TupleDef *getReturnType() { return returnType; }		
        void describe(int indent);
};

class TupleSymbol : public Symbol {
  protected:
	List<Type*> *elementTypes;	
  public:
	TupleSymbol(Identifier *id, TupleDef *def, List<Type*> *elementTypes);
	List<Type*> *getElementTypes() { return elementTypes; }	
        void describe(int indent);
};

class VariableSymbol : public Symbol {
  protected:
	Identifier *id;
	Type *type;
  public:
	VariableSymbol(VariableDef *def);
        VariableSymbol(const char *name, Type *type) : Symbol(name, type) { this->type = type; }
	VariableSymbol(Identifier *id, Type *type, Node *node);	
	void setType(Type *type) { this->type = type; }
	Type *getType() { return type; }	
        void describe(int indent);
};

#endif
