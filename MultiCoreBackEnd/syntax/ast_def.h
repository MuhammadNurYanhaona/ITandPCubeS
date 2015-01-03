#ifndef _H_ast_def
#define _H_ast_def

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "../utils/list.h"
#include "../semantics/scope.h"

class TaskDef;

class Definition : public Node {	
  public:
	Definition() : Node() {}
	Definition(yyltype loc) : Node(loc) {}

	// Each definition, including a variable definition, should have a scope associated with it.
	// Therefore these classes extends from a base class supporting scope attachment after AST
	// is generated for the program. The following methods should be overridden to do the job of
	// scope attachment. Since we do not enforce any ordering between a definition and its first
	// use, scope attachment and validation needs to be done in two different steps. Consequently
	// we have two methods here.
	virtual void attachScope(Scope *parentScope) {} 
	virtual void validateScope(Scope *parentScope) {} 
};

class VariableDef : public Definition {
  protected:
        Identifier *id;
        Type *type;
  public:
        VariableDef(Identifier *id, Type *type);
        VariableDef(Identifier *id);
	const char *GetPrintNameForNode() { return "Variable"; } 
	static List<VariableDef*> *DecomposeDefs(List<Identifier*> *idList, Type *type);
        void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }
	Type *getType() { return type; }
	void setType(Type *type) { this->type = type; }
	void validateScope(Scope *parentScope); 
};

class ProgramDef : public Definition {
  protected:
	List<Node*> *components;
  public:
	static ProgramDef *program;
	ProgramDef(List<Node*> *components);
	const char *GetPrintNameForNode() { return "Program"; } 
	void PrintChildren(int indentLevel);	
	void attachScope(Scope *parentScope); 
	void validateScope(Scope *parentScope);
	void performStaticAnalysis();
	void printTasks();
	Node *getTaskDefinition(const char *taskName); 
	List<TaskDef*> *getTasks(); 
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
	void attachScope(Scope *parentScope); 
	void validateScope(Scope *parentScope);
	List<VariableDef*> *getComponents() { return components; }
};

class CoordinatorDef : public Definition {
  protected:
	Identifier *argument;
	List<Stmt*> *code;
	Scope *executionScope;
  public:
	CoordinatorDef(Identifier *argument, List<Stmt*> *code, yyltype loc);
	const char *GetPrintNameForNode() { return "Main"; } 
	void PrintChildren(int indentLevel);	
	void validateScope(Scope *parentScope);
};

class FunctionHeader : public Node {
  protected:	
	List<VariableDef*> *arguments;
	List<VariableDef*> *results;
  public:
	FunctionHeader(List<VariableDef*> *arguments, List<VariableDef*> *results);
	const char *GetPrintNameForNode() { return "Header"; } 
	void PrintChildren(int indentLevel);
	List<VariableDef*> *getArguments() { return arguments; }
	List<VariableDef*> *getResults() { return results; }	
};

class FunctionDef : public Definition {
  protected:
	Identifier *id;
	FunctionHeader *header;
	List<Stmt*> *code;
	Scope *scope;
  public:
	FunctionDef(Identifier *id, FunctionHeader *header, List<Stmt*> *code);
	const char *GetPrintNameForNode() { return "Function"; } 
	void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }	
	void attachScope(Scope *parentScope); 
	void validateScope(Scope *parentScope);
};

#endif
