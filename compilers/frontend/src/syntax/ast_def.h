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

class Definition : public Node {	
  public:
	Definition() : Node() {}
	Definition(yyltype loc) : Node(loc) {}
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
};

class ProgramDef : public Definition {
  protected:
	List<Node*> *components;
  public:
	static ProgramDef *program;
	ProgramDef(List<Node*> *components);
	const char *GetPrintNameForNode() { return "Program"; } 
	void PrintChildren(int indentLevel);	
};

class TupleDef : public Definition {
  protected:
        Identifier *id;
        List<VariableDef*> *components;
  public:
        TupleDef(Identifier *id, List<VariableDef*> *components); 
	const char *GetPrintNameForNode() { return "Tuple"; } 
        void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }
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
};

class FunctionArg : public Node {
  protected:	
	Identifier *name;
	ArgumentType type;		// tells if reference or value argument
  public:
	FunctionArg(Identifier *name, ArgumentType type);
	const char *GetPrintNameForNode() { return "Argument"; } 
	void PrintChildren(int indentLevel);
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
};

#endif
