#ifndef _H_scope
#define _H_scope

#include "../syntax/ast.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>
#include <fstream>

// We need different scope types as different parts of the source code follows different scoping
// rules.
enum ScopeType {	
			ProgramScope, 
			TupleScope,
			TaskScope, 
			TaskInitScope,
			TaskPartitionScope,
			FunctionScope, 
			StatementBlockScope, 
			ExecutionScope,	

			// This are the scope types that support type inferrences for variables.
			// So when an undeclared variable's type is inferred in these scopes or
			// in any scope nested within one of these scope. The variable gets reg-
			// istered here for declaration during code generation. 
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

	//--------------------------------------------------- Basic methods for scope management
        Scope* enter_scope(Scope *newScope);
        void insert_symbol(Symbol *symbol);
	void remove_symbol(const char *key);
        Symbol* lookup(const char *key);
        Symbol* local_lookup(const char *key);
        Scope* exit_scope();

	//--------------------------------------------------------------------- helper functions
	void detach_from_parent();
	bool insert_inferred_symbol(Symbol *symbol);
	void copy_symbol(Symbol *symbol);
        Iterator<Symbol*> get_local_symbols() { return symbolTable->GetIterator(); }
        void describe(int indent);
	Scope *get_nearest_scope(ScopeType type);

	//------------------------------------------------------- Code Generation Hack Functions
        /***************************************************************************************
          The code generation related function definitions that are placed here are platform 
	  specific. So ideally they should not be included here and the frontend compiler 
	  should be oblivious of them. However, as we ran out of time in overhauling the old 
	  compilers, instead of redesigning the code generation process, we decided to keep 
	  the union of old function definitions in the frontend and put their implementations 
	  in relevent backend compilers.   
        ***************************************************************************************/	
        
	void declareVariables(std::ostringstream &stream, int indent);
};


#endif
