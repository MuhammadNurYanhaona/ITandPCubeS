#ifndef _H_ast_def
#define _H_ast_def

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"

#include <fstream>
#include <sstream>

class TaskDef;
class TupleDef;
class CoordinatorDef;

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
  public:
	FunctionDef(Identifier *id, List<FunctionArg*> *arguments, Stmt *code);
	const char *GetPrintNameForNode() { return "Sequential-Function"; } 
	void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }	
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone();
	DefTypeId getDefTypeId() { return FN_DEF; }
	
	// The reference and value type arguments should be treated differently inside the function body.
	// This function annotates the expressions that involves any argument access with proper type for
	// later analysis.
	void annotateArgAccessesByType();
};

#endif
