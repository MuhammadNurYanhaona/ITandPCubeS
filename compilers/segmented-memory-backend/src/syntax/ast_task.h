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

#include <sstream>

class PartitionSection;
class MetaComputeStage;
class FlowStage;
class CompositeStage;
class VariableAccess;

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
	
	//----------------------------------------------------------------------------------------Syntax Analysis Routines
        const char *GetPrintNameForNode() { return "Initialization Instructions"; }
        void PrintChildren(int indentLevel);
	
	//--------------------------------------------------------------------------------------Semantic Analysis Routines
	List<Type*> *getArgumentTypes() { return argumentTypes; }
	void generateScope(Scope *parentScope);
	
	//----------------------------------------------------------------------------------------Static Analysis Routines
	void performVariableAccessAnalysis(Scope *taskGlobalScope);
	void printUsageStatistics();

	//---------------------------------------------------------------------------Helper routines for back-end compiler
	List<const char*> *getArguments();
	void generateCode(std::ostringstream &stream);
};

class EnvironmentLink : public Node {
  protected:
	Identifier *var;
	LinkageType mode;
  public:
	EnvironmentLink(Identifier *var, LinkageType mode);
	
	//----------------------------------------------------------------------------------------Syntax Analysis Routines
	static List<EnvironmentLink*> *decomposeLinks(List<Identifier*> *ids, LinkageType mode);	
        const char *GetPrintNameForNode();
        void PrintChildren(int indentLevel);
	Identifier *getVariable() { return var; }

	//---------------------------------------------------------------------------Helper routines for back-end compiler
	bool isExternal() { return (mode == TypeLink || mode == TypeCreateIfNotLinked); }
	bool isNullable() { return (mode == TypeCreateIfNotLinked || mode == TypeCreate); }
};

class EnvironmentConfig : public Node {
  protected:
	List<EnvironmentLink*> *links;
  public:
	EnvironmentConfig(List<EnvironmentLink*> *links, yyltype loc);			
        
	//---------------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Environment Configuration"; }
        void PrintChildren(int indentLevel);
	List<EnvironmentLink*> *getLinks() { return links; }
	
	//-------------------------------------------------------------------------------------Semantic Analysis Routines
	void attachScope();
};

//-----------------------------------------------------------------------------------------------------------------------
/* This version of Data Flow Stage is no longer in use. We keep it around for now as it has some 
   utility routines that are used for creating the flow stages mentioned in data_flow.h header file.
   TODO we should get rid of this class after moving the remaining one or two useful functions in
   other places.	
*/
class DataFlowStage : public Node {
  public:
        int nestingIndex;
        int computeIndex;
        Space *executionSpace;
        Hashtable<VariableAccess*> *accessMap;
        DataDependencies *dataDependencies;
        DataFlowStage *nestingController;
        DataFlowStage(yyltype loc);
        void setNestingIndex(int nestingIndex);
        void setNestingController(DataFlowStage *controller);
        int getNestingIndex();
        void setComputeIndex(int computeIndex);
        int getComputeIndex();
        DataDependencies *getDataDependencies();
        virtual Hashtable<VariableAccess*> *getAccessMap();
        Space *getSpace();
        const char *performDependencyAnalysis(List<DataFlowStage*> *stageList);
};
//----------------------------------------------------------------------------------------------------------------------

class StageHeader : public Node {
  protected:
	Identifier *stageId;
	char spaceId;
	Expr *activationCommand;
  public:
	StageHeader(Identifier *stageId, char spaceId, Expr *activationCommand);	
        
	//--------------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Stage Header"; }
        void PrintChildren(int indentLevel);
	
	//------------------------------------------------------------------------------------Semantic Analysis Routines
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	const char *getStageName() { return stageId->getName(); }
	char getSpaceId() { return spaceId; }
	
	//--------------------------------------------------------------------------------------Static Analysis Routines
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
        
	//------------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Computation Stage"; }
        void PrintChildren(int indentLevel);
	const char *getName() { return header->getStageName(); }
	
	//----------------------------------------------------------------------------------Semantic Analysis Routines
	void setExecutionSpaceCap(Space *space) { executionSpaceCap = space; }
	void validateScope(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setRepeatLoopSpace(Space *space);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);

	//------------------------------------------------------------------------------------Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	int assignFlowStageAndNestingIndexes(int currentNestingIndex, 
			int currentStageIndex, List<DataFlowStage*> *currentStageList);
	Hashtable<VariableAccess*> *getAccessMap();
	void constructComputationFlow(List<FlowStage*> *inProgressStageList, 
			CompositeStage *currentContainerStage);
	
	//-----------------------------------------------------------------------Helper routines for back-end compiler
	void populateRepeatIndexes(List <const char*> *currentList);
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
        
	//-----------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Repeat Control"; }
        void PrintChildren(int indentLevel);
	
	//---------------------------------------------------------------------------------Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpaceCap(Space *space, PartitionHierarchy *partitionHierarchy);
	Space *getExecutionSpace(PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space, PartitionHierarchy *partitionHierarchy);
	const char *getFirstComputeStageName() { return begin->getName(); }

	//-----------------------------------------------------------------------------------Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	bool iterateSubpartitions() { return subpartitionIteration; }
	void changeBeginning(Identifier *newBeginning) { begin = newBeginning; }
	Expr *getCondition() { return rangeExpr; }
	
	//----------------------------------------------------------------------Helper routines for back-end compiler
	void populateRepeatIndexes(List <const char*> *currentList);
};

class MetaComputeStage : public Node {
  protected:
	List<ComputeStage*> *stageSequence;
	RepeatControl *repeatInstr;
	Space *executionSpace;
  public:
	MetaComputeStage(List<ComputeStage*> *stageSequence, RepeatControl *repeatInstr);	
        
	//-----------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Stage Sequence"; }
        void PrintChildren(int indentLevel);
	
	//---------------------------------------------------------------------------------Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	void setExecutionSpace(Space *space) {  executionSpace = space; }
	void setRepeatLoopSpace(Space *space);

	//-----------------------------------------------------------------------------------Static Analysis Routines
	void checkVariableAccess(Scope *taskGlobalScope);
	int assignFlowStageAndNestingIndexes(int currentNestingIndex, 
			int currentStageIndex, List<DataFlowStage*> *currentStageList);
	Hashtable<VariableAccess*> *getAggregateAccessMapOfNestedStages();
	void constructComputationFlow(List<FlowStage*> *inProgressStageList, 
			CompositeStage *currentContainerStage);

	//----------------------------------------------------------------------Helper routines for back-end compiler
	void populateRepeatIndexes(List <const char*> *currentList);
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
        
	//----------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Compute Section"; }
        void PrintChildren(int indentLevel);
	
	//--------------------------------------------------------------------------------Semantic Analysis Routines
	void validateScopes(Scope *rootScope, PartitionHierarchy *partitionHierarchy);
	
	//----------------------------------------------------------------------------------Static Analysis Routines
	void performVariableAccessAnalysis(Scope *taskGlobalScope);
	void assignFlowStageAndNestingIndexes(List<DataFlowStage*> *currentStageList);
	// This recursively construct the control flow of the compute block by calling the same method in 
	// meta-compute and compute stages. The layout of the compute section after abstract syntax tree been
	// generated is not particularly suitable for later analysis and thereby proper code generation.
	// Henceforth we include this mechanism to transform the content of the compute section into a new
	// form that is more appropriate for later stages.
	void constructComputationFlow(Space *rootSpace);
	// This is again a recursive process of determining the dependency arcs among flow stages
	void performDependencyAnalysis(PartitionHierarchy *hierarchy);
	void print();

	//---------------------------------------------------------------------Helper functions for backend compiler
	List<const char*> *getRepeatIndexes();
	CompositeStage *getComputation();
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
        
	//---------------------------------------------------------------------------------Syntax Analysis Routines
	const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);

	//-------------------------------------------------------------------------------Semantic Analysis Routines
	void attachScope(Scope *parentScope);
	void validateScope(Scope *parentScope);
	TupleDef *getEnvTuple() { return envTuple; }
	TupleDef *getPartitionTuple() { return partitionTuple; }
	List<Type*> *getInitArgTypes();
	int getPartitionArgsCount();
	DefineSection *getDefineSection() { return define; }	
	void typeCheckInitSection(Scope *scope);
	void constructPartitionHierarchy();
	void validateComputeSection(Scope *rootScope);

	//--------------------------------------------------------------------------------Static Analysis Routines
	// AnalyseCode method does a serious of static analysis of the code such as dependency analysis,
	// synchronization requirements determination, and so on to prepare the intermediate representation
	// for back-end code generation. In this process it translates the compute section into a new 
	// recursive compute+data flow form that is easier to handle for later phases of the compiler.
	void analyseCode();
	// This function returns information regarding the nature of access of all environmental data 
	// structures of the task. It must be called after the analyseCode() function has been invoked
	// as that function generates, among other things, per compute stage data access information.
	// This function just accumulates information of selective data structures from the statistics
	// generated by the earlier function. 
	List<VariableAccess*> *getAccessLogOfEnvVariables();
	void print();

	//-----------------------------------------------------------------helper functions for back-end compilers
	const char *getName() { return id->getName(); }
	PartitionHierarchy *getPartitionHierarchy();
	List<Identifier*> *getPartitionArguments();
	List<EnvironmentLink*> *getEnvironmentLinks() { return environment->getLinks(); }
	List<const char*> *getRepeatIndexes() { return compute->getRepeatIndexes(); }
	InitializeInstr *getInitSection() { return initialize; }
	CompositeStage *getComputation();
	EnvironmentLink *getEnvironmentLink(const char *linkName);
};

#endif
