#ifndef _H_ast_task
#define _H_ast_task

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "ast_def.h"
#include "ast_expr.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../semantics/task_space.h"
#include "../static-analysis/data_access.h"
#include "../static-analysis/data_flow.h"

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
	Hashtable<VariableAccess*> *accessMap;
  public: 
	InitializeInstr(List<Identifier*> *arguments, List<Stmt*> *code, yyltype loc);	
	
	// Syntax Analysis Routines
        const char *GetPrintNameForNode() { return "Initialization Instructions"; }
        void PrintChildren(int indentLevel);
	
	// Semantic Analysis Routines
	List<Type*> *getArgumentTypes() { return argumentTypes; }
	void generateScope(Scope *parentScope);
	
	// Static Analysis Routines
	void performVariableAccessAnalysis(Scope *taskGlobalScope);
	void printUsageStatistics();
};

class EnvironmentLink : public Node {
  protected:
	Identifier *var;
	LinkageType mode;
  public:
	EnvironmentLink(Identifier *var, LinkageType mode);
	
	// Syntax Analysis Routines
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
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Environment Configuration"; }
        void PrintChildren(int indentLevel);
	List<EnvironmentLink*> *getLinks() { return links; }
	
	// Semantic Analysis Routines
	void attachScope();
};

class StageHeader : public Node {
  protected:
	Identifier *stageId;
	char spaceId;
	Expr *activationCommand;
  public:
	StageHeader(Identifier *stageId, char spaceId, Expr *activationCommand);	
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Stage Header"; }
        void PrintChildren(int indentLevel);
	
	// Semantic Analysis Routines
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	const char *getStageName() { return stageId->getName(); }
	char getSpaceId() { return spaceId; }
	
	// Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope, Hashtable<VariableAccess*> *accessMap);
	Expr *getActivationCommand() { return activationCommand; }
};

class ComputeStage : public DataFlowStage {
  protected:
	StageHeader *header;
	bool metaStage;
	List<Stmt*> *code;
	List<MetaComputeStage*> *nestedSequence;
	Scope *scope;
	Space *executionSpaceCap;
	Space *repeatLoopSpace;
  public:
	ComputeStage(StageHeader *header, List<Stmt*> *code);
	ComputeStage(StageHeader *header, List<MetaComputeStage*> *nestedSequence);
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Computation Stage"; }
        void PrintChildren(int indentLevel);
	const char *getName() { return header->getStageName(); }
	
	// Semantic Analysis Routines
	void setExecutionSpaceCap(Space *space) { executionSpaceCap = space; }
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setRepeatLoopSpace(Space *space);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);

	// Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	int assignFlowStageAndNestingIndexes(int currentNestingIndex, 
			int currentStageIndex, List<DataFlowStage*> *currentStageList);
	Hashtable<VariableAccess*> *getAccessMap();
	void constructComputationFlow(List<FlowStage*> *inProgressStageList, CompositeStage *currentContainerStage);
};

class RepeatControl : public DataFlowStage {
  protected:
	Identifier *begin;
	Expr* rangeExpr;
	Space *executionSpaceCap;
	bool subpartitionIteration;
	bool traversed;
  public:
	RepeatControl(Identifier *begin, Expr *rangeExpr, yyltype loc);
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Repeat Control"; }
        void PrintChildren(int indentLevel);
	
	// Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpaceCap(Space *space, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space, PartitionHierarchy *partitionHierarchy);
	const char *getFirstComputeStageName() { return begin->getName(); }

	// Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	bool iterateSubpartitions() { return subpartitionIteration; }
	void changeBeginning(Identifier *newBeginning) { begin = newBeginning; }
	Expr *getCondition() { return rangeExpr; }
};

class MetaComputeStage : public Node {
  protected:
	List<ComputeStage*> *stageSequence;
	RepeatControl *repeatInstr;
	Space *executionSpace;
  public:
	MetaComputeStage(List<ComputeStage*> *stageSequence, RepeatControl *repeatInstr);	
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Stage Sequence"; }
        void PrintChildren(int indentLevel);
	
	// Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space) {  executionSpace = space; }
	void setRepeatLoopSpace(Space *space);

	// Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	int assignFlowStageAndNestingIndexes(int currentNestingIndex, 
			int currentStageIndex, List<DataFlowStage*> *currentStageList);
	Hashtable<VariableAccess*> *getAggregateAccessMapOfNestedStages();
	void constructComputationFlow(List<FlowStage*> *inProgressStageList, CompositeStage *currentContainerStage);
};

class ComputeSection : public Node {
  protected:
	// This represents the content of the compute block of a task as it retrieved during abstract syntax
	// tree generation phase.
	List<MetaComputeStage*> *stageSeqList;
	// This represents the final outlook of the compute block as a graph of data flow stages at the end
	// of the static analysis phase.
	CompositeStage *computation;
  public:
	ComputeSection(List<MetaComputeStage*> *stageSeqList, yyltype loc);		
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Compute Section"; }
        void PrintChildren(int indentLevel);
	
	// Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	
	// Static Analysis Routines
	void performVariableAccessAnalysis(Scope *taskGlobalScope);
	void assignFlowStageAndNestingIndexes(List<DataFlowStage*> *currentStageList);
	// This recursively construct the control flow of the compute block by calling the same method in 
	// meta-compute and compute stages.
	void constructComputationFlow(Space *rootSpace);
	// This is again a recursive process of determining the dependency arcs among flow stages
	void performDependencyAnalysis(PartitionHierarchy *hierarchy) { 
		computation->performDependencyAnalysis(hierarchy); 
	}
	void print() { computation->print(0); }
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
        
	// Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);

	// Semantic Analysis Routines
	void attachScope(Scope *parentScope);
	void validateScope(Scope *parentScope);
	TupleDef *getEnvTuple() { return envTuple; }
	List<Type*> *getInitArgTypes();
	int getPartitionArgsCount();
	DefineSection *getDefineSection() { return define; }	
	void typeCheckInitSection(Scope *scope);
	void constructPartitionHierarchy();
	void validateComputeSection(Scope *rootScope);

	// Static Analysis Routines
	void analyseCode();
	void print();

	// helper functions for back-end compilers
	const char *getName() { return id->getName(); }
	PartitionHierarchy *getPartitionHierarchy();
	List<Identifier*> *getPartitionArguments();
};

#endif
