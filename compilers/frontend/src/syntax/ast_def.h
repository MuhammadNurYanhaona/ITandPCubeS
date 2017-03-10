#ifndef _H_ast_def
#define _H_ast_def

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <fstream>
#include <sstream>

class TaskDef;
class TupleDef;
class CoordinatorDef;
class FunctionInstance;

enum DefTypeId { VAR_DEF, PROG_DEF, FN_DEF, CLASS_DEF, COORD_DEF, TASK_DEF };

class Definition : public Node {	
  public:
	Definition() : Node() {}
	Definition(yyltype loc) : Node(loc) {}
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	// Each definition, including a variable definition, should have a scope associated with it.
        // Therefore these classes extend from a base class supporting scope attachment after AST
        // is generated for the program. The following methods should be overridden to do the job of
        // scope attachment. Since we do not enforce any ordering between a definition and its first
        // use, scope attachment and validation needs to be done in two different steps for some
	// definition types. Consequently, we have two methods here.
        virtual void attachScope(Scope *parentScope) {}
        virtual void validateScope(Scope *parentScope) {}

	// function needed to filter different types of definitions; each subclass should return a
	// unique type ID
	virtual DefTypeId getDefTypeId() = 0;
};

class VariableDef : public Definition {
  protected:
        Identifier *id;
        Type *type;

	// This indicates if the current variable is meant to be used as the result of some reduction
        // operation. If YES then the storage and access of this variables need a different treatment
        // from other task-global variables.  
        bool reduction;
  public:
        VariableDef(Identifier *id, Type *type);
        VariableDef(Identifier *id);
	const char *GetPrintNameForNode() { return "Variable"; } 
	static List<VariableDef*> *DecomposeDefs(List<Identifier*> *idList, Type *type);
        void PrintChildren(int indentLevel);
	void flagAsReduction() { this->reduction = true; }
        bool isReduction() { return reduction; }
	Identifier *getId() { return id; }
	Type *getType() { return type; }
	void setType(Type *type) { this->type = type; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	void validateScope(Scope *parentScope);
	DefTypeId getDefTypeId() { return VAR_DEF; }
};

class ProgramDef : public Definition {
  protected:
	List<Node*> *components;
  public:
	static ProgramDef *program;
	ProgramDef(List<Node*> *components);
	const char *GetPrintNameForNode() { return "Program"; } 
	void PrintChildren(int indentLevel);	
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	DefTypeId getDefTypeId() { return PROG_DEF; }
	
	// the function that encapsulates the entire semantic analysis phase
	void performScopeAndTypeChecking();

	// component functions for doing different steps of semantic analysis
	void analyseCustomTypes();

	List<Definition*> *getComponentsByType(DefTypeId typeId);
};

class TupleDef : public Definition {
  protected:
        Identifier *id;
        List<VariableDef*> *components;

	// This is a flag to indicate if this custom type represents the environment of some task. 
	// We need this to do specialized operations needed by environment objects. For example, 
	// we declare pointers for environment types as opposed to object instances done for other 
        // tuple types.
        bool environment;
  public:
        TupleDef(Identifier *id, List<VariableDef*> *components); 
	const char *GetPrintNameForNode() { return "Class"; } 
        void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	void attachScope(Scope *parentScope);
        void validateScope(Scope *parentScope);
	DefTypeId getDefTypeId() { return CLASS_DEF; }
	void flagAsEnvironment() { environment = true; }
        bool isEnvironment() { return environment; }
	VariableDef *getComponent(const char *name);
};

class CoordinatorDef : public Definition {
  protected:
	Identifier *argument;
	List<Stmt*> *code;
	TupleDef *argumentTuple;
  public:
	CoordinatorDef(Identifier *argument, List<Stmt*> *code, yyltype loc);
	const char *GetPrintNameForNode() { return "Program-Coordinator"; } 
	void PrintChildren(int indentLevel);	
	TupleDef *getArgumentTuple() { return argumentTuple; }
	const char *getArgumentName() { return argument->getName(); }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	DefTypeId getDefTypeId() { return COORD_DEF; }
};

class FunctionArg : public Node {
  protected:	
	Identifier *name;
	ArgumentType type;		// tells if reference or value argument
  public:
	FunctionArg(Identifier *name, ArgumentType type);
	const char *GetPrintNameForNode() { return "Argument"; } 
	void PrintChildren(int indentLevel);
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	Identifier *getName() { return name; }
	ArgumentType getType() { return type; }
};

class FunctionDef : public Definition {
  protected:
	Identifier *id;
	List<FunctionArg*> *arguments;
	Stmt *code;
	List<FunctionInstance*> *instanceList;
  public:
	FunctionDef(Identifier *id, List<FunctionArg*> *arguments, Stmt *code);
	const char *GetPrintNameForNode() { return "Sequential-Function"; } 
	void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }	
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	// a static map of function definitions is needed to retrieve a matching function definition by
	// a matching name during polymorphic function resolutions based on call context
	static Hashtable<FunctionDef*> *fnDefMap;
	
	Node *clone();
	DefTypeId getDefTypeId() { return FN_DEF; }
	List<FunctionArg*> *getArguments() { return arguments; }
	Stmt *getCode() { return code; }
	
	// The reference and value type arguments should be treated differently inside the function body.
	// This function annotates the expressions that involves any argument access with proper type for
	// later analysis.
	void annotateArgAccessesByType();

	// This creates a new function instance for a specific parameter types combination that arouse from
	// a specific call context. this works if the type resolution process on the function body completes 
	// successfully of course. 
	Type *resolveFnInstanceForParameterTypes(Scope *programScope, 
			List<Type*> *paramTypes, 
			Identifier *callerId);
  protected:
	FunctionInstance *getInstanceForParamTypes(List<Type*> *paramTypes);	
};

// IT functions are type polymorphic. To generate code for a function for a specific call context, we need
// to resolve the function body for the parameter types and generate a type-specific copy of the function.
// This class represents such a type-specific instance. We could not just generate a templated function from
// original function definition as IT uses type inference to resolve local variables' types and not fixing
// the return types of functions would leave any expressions having a function call also unresolved. 
class FunctionInstance {
  protected:
	const char *fnName;
	List<FunctionArg*> *arguments;
	List<Type*> *argumentTypes;
	Type *returnType;
	Stmt *code;
  public:
	FunctionInstance(FunctionDef *fnDef, 
		int instanceId, List<Type*> *argTypes, Scope *programScope);
	Type *getReturnType() { return returnType; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void performScopeAndTypeChecking(Scope *programScope);
	bool isMatchingArguments(List<Type*> *argTypeList);
};

#endif
