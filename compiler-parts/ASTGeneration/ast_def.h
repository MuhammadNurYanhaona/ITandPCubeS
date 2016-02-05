#ifndef _H_ast_def
#define _H_ast_def

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "list.h"

class VariableDef : public Node {
  protected:
        Identifier *id;
        Type *type;
  public:
        VariableDef(Identifier *id, Type *type);
	const char *GetPrintNameForNode() { return "Variable"; } 
	static List<VariableDef*> *DecomposeDefs(List<Identifier*> *idList, Type *type);
        void PrintChildren(int indentLevel);
};

class ProgramDef : public Node {
  protected:
	List<Node*> *components;
  public:
	ProgramDef(List<Node*> *components);
	const char *GetPrintNameForNode() { return "Program"; } 
	void PrintChildren(int indentLevel);	
};

class TupleDef : public Node {
  protected:
        Identifier *id;
        List<VariableDef*> *components;
  public:
        TupleDef(Identifier *id, List<VariableDef*> *components); 
	const char *GetPrintNameForNode() { return "Tuple"; } 
        void PrintChildren(int indentLevel);
};

class CoordinatorDef : public Node {
  protected:
	Identifier *argument;
	List<Stmt*> *code;
  public:
	CoordinatorDef(Identifier *argument, List<Stmt*> *code, yyltype loc);
	const char *GetPrintNameForNode() { return "Main"; } 
	void PrintChildren(int indentLevel);	
};

class FunctionHeader : public Node {
  protected:	
	List<VariableDef*> *arguments;
	List<VariableDef*> *results;
  public:
	FunctionHeader(List<VariableDef*> *arguments, List<VariableDef*> *results);
	const char *GetPrintNameForNode() { return "Header"; } 
	void PrintChildren(int indentLevel);	
};

class FunctionDef : public Node {
  protected:
	Identifier *id;
	FunctionHeader *header;
	List<Stmt*> *code;
  public:
	FunctionDef(Identifier *id, FunctionHeader *header, List<Stmt*> *code);
	const char *GetPrintNameForNode() { return "Function"; } 
	void PrintChildren(int indentLevel);	
};

#endif
