#ifndef _H_scope
#define _H_scope

#include "../syntax/ast.h"
#include "../utils/hashtable.h"

enum ScopeType {	
			ProgramScope, 
			TupleScope,
			TaskScope, 
			TaskInitScope,
			TaskPartitionScope,
			FunctionScope, 
			StatementBlockScope, 
			ExecutionScope,	
 
			TaskInitBodyScope,
			FunctionBodyScope,
			ComputationStageScope, 
			CoordinatorScope
};

class Symbol;

class Scope {
    protected:
        Scope *parent;
        ScopeType type;
        Hashtable<Symbol*> *symbolTable;
	bool supportTypeInference;

    public:

        Scope(ScopeType type);

        Scope* enter_scope(Scope *newScope);
        void insert_symbol(Symbol *symbol);
	void remove_symbol(const char *key);
        Symbol* lookup(const char *key);
        Symbol* local_lookup(const char *key);
        Scope* exit_scope();

	void detach_from_parent();
	bool insert_inferred_symbol(Symbol *symbol);
	void copy_symbol(Symbol *symbol);
        Iterator<Symbol*> get_local_symbols() { return symbolTable->GetIterator(); }
        void describe(int indent);
	Scope *get_nearest_scope(ScopeType type);
};


#endif
