#ifndef _H_ast_task
#define _H_ast_task

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "ast_def.h"
#include "ast_expr.h"
#include "ast_partition.h"
#include "list.h"

class MetaComputeStage;

enum LinkageType { TypeCreate, TypeLink, TypeCreateIfNotLinked };

class DefineSection : public Node {
  protected:
	List<VariableDef*> *define;
  public:
	DefineSection(List<VariableDef*> *def, yyltype loc);
        const char *GetPrintNameForNode() { return "Variable Declarations"; }
        void PrintChildren(int indentLevel);
};

class InitializeInstr : public Node {
  protected:
	List<Identifier*> *arguments;
	List<Stmt*> *code;
  public: 
	InitializeInstr(List<Identifier*> *arguments, List<Stmt*> *code, yyltype loc);	
        const char *GetPrintNameForNode() { return "Initialization Instructions"; }
        void PrintChildren(int indentLevel);
};

class EnvironmentLink : public Node {
  protected:
	Identifier *var;
	LinkageType mode;
  public:
	EnvironmentLink(Identifier *var, LinkageType mode);
	static List<EnvironmentLink*> *decomposeLinks(List<Identifier*> *ids, LinkageType mode);	
        const char *GetPrintNameForNode();
        void PrintChildren(int indentLevel);
};

class EnvironmentConfig : public Node {
  protected:
	List<EnvironmentLink*> *links;
  public:
	EnvironmentConfig(List<EnvironmentLink*> *links, yyltype loc);			
        const char *GetPrintNameForNode() { return "Environment Configuration"; }
        void PrintChildren(int indentLevel);
};

class StageHeader : public Node {
  protected:
	Identifier *stageId;
	char spaceId;
	Expr *activationCommand;
  public:
	StageHeader(Identifier *stageId, char spaceId, Expr *activationCommand);	
        const char *GetPrintNameForNode() { return "Stage Header"; }
        void PrintChildren(int indentLevel);
};

class ComputeStage : public Node {
  protected:
	StageHeader *header;
	bool metaStage;
	List<Stmt*> *code;
	List<MetaComputeStage*> *nestedSequence;
  public:
	ComputeStage(StageHeader *header, List<Stmt*> *code);
	ComputeStage(StageHeader *header, List<MetaComputeStage*> *nestedSequence);
        const char *GetPrintNameForNode() { return "Computation Stage"; }
        void PrintChildren(int indentLevel);
};

class RepeatControl : public Node {
  protected:
	Identifier *begin;
	Expr* rangeExpr;
  public:
	RepeatControl(Identifier *begin, Expr *rangeExpr, yyltype loc);
        const char *GetPrintNameForNode() { return "Repeat Control"; }
        void PrintChildren(int indentLevel);
};

class MetaComputeStage : public Node {
  protected:
	List<ComputeStage*> *stageSequence;
	RepeatControl *repeatInstr;
  public:
	MetaComputeStage(List<ComputeStage*> *stageSequence, RepeatControl *repeatInstr);	
        const char *GetPrintNameForNode() { return "Stage Sequence"; }
        void PrintChildren(int indentLevel);
};

class ComputeSection : public Node {
  protected:
	List<MetaComputeStage*> *stageSeqList;
  public:
	ComputeSection(List<MetaComputeStage*> *stageSeqList, yyltype loc);		
        const char *GetPrintNameForNode() { return "Compute Section"; }
        void PrintChildren(int indentLevel);
};

class TaskDef : public Node {
  protected:
        Identifier *id;
	DefineSection *define;
	EnvironmentConfig *environment;
	InitializeInstr *initialize;
	ComputeSection	*compute;
	PartitionSection *partition;
  public:
        TaskDef(Identifier *id, DefineSection *define, EnvironmentConfig *environment, 
		InitializeInstr *initialize, ComputeSection *compute, PartitionSection *partition);
        const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);
};

#endif
