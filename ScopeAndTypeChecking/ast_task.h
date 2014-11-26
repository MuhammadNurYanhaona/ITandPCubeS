#ifndef _H_ast_task
#define _H_ast_task

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "ast_def.h"
#include "ast_expr.h"
#include "list.h"
#include "task_space.h"

class PartitionSection;
class MetaComputeStage;

enum LinkageType { TypeCreate, TypeLink, TypeCreateIfNotLinked };

class DefineSection : public Node {
  protected:
	List<VariableDef*> *define;
  public:
	DefineSection(List<VariableDef*> *def, yyltype loc);
        const char *GetPrintNameForNode() { return "Variable Declarations"; }
        void PrintChildren(int indentLevel);
	List<VariableDef*> *getDefinitions() { return define; }
};

class InitializeInstr : public Node {
  protected:
	List<Identifier*> *arguments;
	List<Type*> *argumentTypes;
	List<Stmt*> *code;
	Scope *scope;
  public: 
	InitializeInstr(List<Identifier*> *arguments, List<Stmt*> *code, yyltype loc);	
        const char *GetPrintNameForNode() { return "Initialization Instructions"; }
        void PrintChildren(int indentLevel);
	List<Type*> *getArgumentTypes() { return argumentTypes; }
	void generateScope(Scope *parentScope);
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
	Identifier *getVariable() { return var; }
};

class EnvironmentConfig : public Node {
  protected:
	List<EnvironmentLink*> *links;
  public:
	EnvironmentConfig(List<EnvironmentLink*> *links, yyltype loc);			
        const char *GetPrintNameForNode() { return "Environment Configuration"; }
        void PrintChildren(int indentLevel);
	void attachScope();
	List<EnvironmentLink*> *getLinks() { return links; }
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
	
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	const char *getStageName() { return stageId->getName(); }
};

class ComputeStage : public Node {
  protected:
	StageHeader *header;
	bool metaStage;
	List<Stmt*> *code;
	List<MetaComputeStage*> *nestedSequence;
	Scope *scope;
	Space *executionSpaceCap;
	Space *executionSpace;
	Space *repeatLoopSpace;
  public:
	ComputeStage(StageHeader *header, List<Stmt*> *code);
	ComputeStage(StageHeader *header, List<MetaComputeStage*> *nestedSequence);
        
	const char *GetPrintNameForNode() { return "Computation Stage"; }
        void PrintChildren(int indentLevel);
	const char *getName() { return header->getStageName(); }
	
	void setExecutionSpaceCap(Space *space) { executionSpaceCap = space; }
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setRepeatLoopSpace(Space *space) { repeatLoopSpace = space; }
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
};

class RepeatControl : public Node {
  protected:
	Identifier *begin;
	Expr* rangeExpr;
	Space *executionSpaceCap;
	Space *executionSpace;
	bool subpartitionIteration;
  public:
	RepeatControl(Identifier *begin, Expr *rangeExpr, yyltype loc);
        
	const char *GetPrintNameForNode() { return "Repeat Control"; }
        void PrintChildren(int indentLevel);
	
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpaceCap(Space *space, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space, PartitionHierarchy *partitionHierarchy);
	const char *getFirstComputeStageName() { return begin->getName(); }
};

class MetaComputeStage : public Node {
  protected:
	List<ComputeStage*> *stageSequence;
	RepeatControl *repeatInstr;
	Space *executionSpace;
  public:
	MetaComputeStage(List<ComputeStage*> *stageSequence, RepeatControl *repeatInstr);	
        
	const char *GetPrintNameForNode() { return "Stage Sequence"; }
        void PrintChildren(int indentLevel);
	
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space) {  executionSpace = space; }
};

class ComputeSection : public Node {
  protected:
	List<MetaComputeStage*> *stageSeqList;
  public:
	ComputeSection(List<MetaComputeStage*> *stageSeqList, yyltype loc);		
        
	const char *GetPrintNameForNode() { return "Compute Section"; }
        void PrintChildren(int indentLevel);
	
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
};

class TaskDef : public Definition {
  protected:
        Identifier *id;
	DefineSection *define;
	EnvironmentConfig *environment;
	InitializeInstr *initialize;
	ComputeSection	*compute;
	PartitionSection *partition;
	TupleDef *envTuple;
	TupleDef *partitionTuple; 
  public:
        TaskDef(Identifier *id, DefineSection *define, EnvironmentConfig *environment, 
		InitializeInstr *initialize, ComputeSection *compute, PartitionSection *partition);
        
	const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);

	void attachScope(Scope *parentScope);
	void validateScope(Scope *parentScope);
	TupleDef *getEnvTuple() { return envTuple; }
	List<Type*> *getInitArgTypes();
	int getPartitionArgsCount();
	DefineSection *getDefineSection() { return define; }	
	void typeCheckInitSection(Scope *scope);
	void constructPartitionHierarchy();
	void validateComputeSection(Scope *rootScope);
};

#endif
