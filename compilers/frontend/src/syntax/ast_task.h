#ifndef _H_ast_task
#define _H_ast_task

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "ast_def.h"
#include "ast_expr.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

class TaskDef;
class PartitionSection;
class CompositeStage;
class ParamReplacementConfig;
class FlowStageConstrInfo;
class VariableAccess;
class TaskEnvStat;
class IncludesAndLinksMap;

class DefineSection : public Node {
  protected:
	List<VariableDef*> *define;
  public:
	DefineSection(List<VariableDef*> *def, yyltype loc);
        const char *GetPrintNameForNode() { return "Define-Section"; }
        void PrintChildren(int indentLevel);
	List<VariableDef*> *getDefinitions() { return define; }
};

class InitializeSection : public Node {
  protected:
	List<Identifier*> *arguments;
	List<Type*> *argumentTypes;
	StmtBlock *code;

	// The Initialize Section has its own scope for executing the code embedded within it 
	Scope *scope;
	// an access map is needed to track what task global variable has been used how during initialization
	Hashtable<VariableAccess*> *accessMap;
  public: 
	InitializeSection(List<Identifier*> *arguments, List<Stmt*> *code, yyltype loc);	
        const char *GetPrintNameForNode() { return "Initialize-Section"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	void performScopeAndTypeChecking(Scope *parentScope);
	List<Type*> *getArgumentTypes() { return argumentTypes; }
	void performVariableAccessAnalysis(Scope *taskGlobalScope);	
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	List<const char*> *getArguments();
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
	LinkageType getMode() { return mode; }
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	bool isExternal() { return (mode == TypeLink || mode == TypeCreateIfNotLinked); }
        bool isNullable() { return (mode == TypeCreateIfNotLinked || mode == TypeCreate); }
};

class EnvironmentSection : public Node {
  protected:
	List<EnvironmentLink*> *links;
  public:
	EnvironmentSection(List<EnvironmentLink*> *links, yyltype loc);			
	const char *GetPrintNameForNode() { return "Environment-Section"; }
        void PrintChildren(int indentLevel);
	List<EnvironmentLink*> *getLinks() { return links; }
};

class StageDefinition : public Node {
  protected:
	Identifier *name;
	List<Identifier*> *parameters;
	Stmt *codeBody;
  public:
	StageDefinition(Identifier *name, List<Identifier*> *parameters, Stmt *codeBody);
	const char *GetPrintNameForNode() { return "Stage-Definition"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	void determineArrayDimensions();
	const char *getName() { return name->getName(); }
	List<Identifier*> *getParameters() { return parameters; }
	Stmt *getCode() { return codeBody; }
};

class StagesSection : public Node {
  protected:
	List<StageDefinition*> *stages;
  public:
	StagesSection(List<StageDefinition*> *stages, yyltype loc);
	const char *GetPrintNameForNode() { return "Stages-Section"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	List<StageDefinition*> *getStageDefinitions() { return stages; }
	StageDefinition *retrieveStage(const char *stageName);	
};

class FlowPart : public Node {
  protected:
	int index;
  public:
	static int currentFlowIndex;
	FlowPart(yyltype loc);
	static void resetFlowIndexRef();

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	// This is the interface for a recursive process of computation flow construction from the
	// abstract syntax tree. The computation flow generated through this process is the primary
	// ingradient for the intermediate representation. Over this flow, all static analyses are 
	// done. Then code generation for different back-end compilers is also done by translating 
	// the flow.
	virtual void constructComputeFlow(CompositeStage *currCompStage,
		FlowStageConstrInfo *cnstrInfo) = 0;	
};

class StageInvocation : public FlowPart {
  protected:
	Identifier *stageName;
	List<Expr*> *arguments;
  public:
	StageInvocation(Identifier *stageName, List<Expr*> *arguments, yyltype loc);
	const char *GetPrintNameForNode() { return "Stage-Invocation"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void constructComputeFlow(CompositeStage *currCompStage, FlowStageConstrInfo *cnstrInfo);

	// This function investigates the argument expressions and the parameters of the compute
	// stage being invoked and determines the best way to convert any use of a parameter in the
	// stage definition body with that of the corresponding argument during the polymorphic type
	// resolution process.
	List<ParamReplacementConfig*> *generateParamReplacementConfigs(); 

	// Some arguments may need to be evaluated into local variables before the execution of the 
	// actual code of the compute-stage. This function generates statements for this operation.
	List<Stmt*> *produceParamGeneratorCode(Scope *stageScope,
			List<ParamReplacementConfig*> *paramReplConfigList);	
};

class CompositeFlowPart : public FlowPart {
  protected:
	List<FlowPart*> *nestedSubflow;
  public:
	CompositeFlowPart(yyltype loc, List<FlowPart*> *nestedSubflow);
        virtual void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	virtual void constructComputeFlow(CompositeStage *currCompStage, 
			FlowStageConstrInfo *cnstrInfo);	
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	virtual void retrieveRepeatIndexes(List <const char*> *currentList);
};

class LpsTransition : public CompositeFlowPart {
  protected:
	char lpsId;
  public:
	LpsTransition(char lpsId, List<FlowPart*> *nestedSubflow, yyltype loc);
	const char *GetPrintNameForNode() { return "LPS-Transition"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void constructComputeFlow(CompositeStage *currCompStage, 
			FlowStageConstrInfo *cnstrInfo);	
};

class ConditionalFlowBlock : public CompositeFlowPart {
  protected:
	Expr *conditionExpr;
  public:
	ConditionalFlowBlock(Expr *conditionExpr, List<FlowPart*> *nestedSubflow, yyltype loc);	
	const char *GetPrintNameForNode() { return "Conditional-Subflow"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void constructComputeFlow(CompositeStage *currCompStage, 
			FlowStageConstrInfo *cnstrInfo);	
};

class EpochBlock : public CompositeFlowPart {
  public:
	EpochBlock(List<FlowPart*> *nestedSubflow, yyltype loc);	
	const char *GetPrintNameForNode() { return "Epoch-Boundary"; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void constructComputeFlow(CompositeStage *currCompStage, 
			FlowStageConstrInfo *cnstrInfo);	
};

class RepeatControl : public Node {
  public:
	RepeatControl(yyltype loc) : Node(loc) {}

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	virtual RepeatCycleType getType() = 0;
	virtual Expr *getRepeatCondition() = 0;
	virtual int validateScopeAndTypes(Scope *executionScope) = 0;
};

class WhileRepeat : public RepeatControl {
  protected:
	Expr *condition;
  public:
	WhileRepeat(Expr *condition, yyltype loc);	
	const char *GetPrintNameForNode() { return "Condition-Traveral"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	RepeatCycleType getType()  { return Condition_Repeat; }
	Expr *getRepeatCondition() { return condition; }

	// do scope-and-type validation of any associated expression and return the error count
	int validateScopeAndTypes(Scope *executionScope);
};

class SubpartitionRepeat : public RepeatControl {
  public:
	SubpartitionRepeat(yyltype loc) : RepeatControl(loc) {}
	const char *GetPrintNameForNode() { return "Subpartition-Traversal"; }
        void PrintChildren(int indentLevel) {}

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	RepeatCycleType getType()  { return Subpartition_Repeat; }
	Expr *getRepeatCondition() { return NULL; }
	int validateScopeAndTypes(Scope *executionScope) { return 0; }
};

class ForRepeat : public RepeatControl {
  protected:
	RangeExpr *rangeExpr;
  public:
	ForRepeat(RangeExpr *rangeExpr, yyltype loc);
	const char *GetPrintNameForNode() { return "Range-Traversal"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	RepeatCycleType getType()  { return Range_Repeat; }
	Expr *getRepeatCondition() { return rangeExpr; }
	int validateScopeAndTypes(Scope *executionScope);
};

class RepeatCycle : public CompositeFlowPart {
  protected:
	RepeatControl *control;
  public:
	RepeatCycle(RepeatControl *control, List<FlowPart*> *nestedSubflow, yyltype loc);
	const char *GetPrintNameForNode() { return "Repeat-Cycle"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void constructComputeFlow(CompositeStage *currCompStage, 
			FlowStageConstrInfo *cnstrInfo);	
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	void retrieveRepeatIndexes(List <const char*> *currentList);
};

class ComputationSection : public Node {
  protected:
	List<FlowPart*> *computeFlow;
  public:
	ComputationSection(List<FlowPart*> *computeFlow, yyltype loc);
	const char *GetPrintNameForNode() { return "Computation-Section"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	// This function produces a computation flow by combining the composite stages from the Stages
	// Section and the instructions this class contains. The resulting flow is returned wrapped 
	// within the composite stage and used for all subsequent static analysis of the task and also
	// for back-end specific code generation.
	CompositeStage *generateComputeFlow(FlowStageConstrInfo *cnstrInfo);
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	// This function retrieves all repeat loop index variables from the computation flow. Repeat
	// loop indexes can be stored separately in each PPU controller without any synchronization
	// with the others. This function is needed to identify these indexes to give them a special 
	// treatment
	void retrieveRepeatIndexes(List <const char*> *currentList);
};

class TaskDef : public Definition {
  protected:
        Identifier *id;
	DefineSection *define;
	EnvironmentSection *environment;
	InitializeSection *initialize;
	StagesSection *stages;
	ComputationSection *compute;
	PartitionSection *partition;

	// the compiler derives automatic custom types to hold environment and partition related properties
	TupleDef *envTuple;
        TupleDef *partitionTuple;

	// the property representing the computation flow of the task that is constructed from information
	// found in different task sections and serves as the intermediate representation for the task
	CompositeStage *computation;
  public:
        TaskDef(Identifier *id, 
		DefineSection *define, 
		EnvironmentSection *environment, 
		InitializeSection *initialize,
		StagesSection *stages, 
		ComputationSection *compute, 
		PartitionSection *partition);
	const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	DefineSection *getDefineSection() { return define; }
	DefTypeId getDefTypeId() { return TASK_DEF; }
	StagesSection *getStagesSection() { return stages; }
	
	// This function is needed to annotate a compute stage's expressions with proper markers before a
	// stage instanciation takes place. For example, we need to analyze the dimensionalities of arrays
	// used within a compute stage definition before we do the type checking and scoping for an
	// invocation of that stage from the Computation Section.
	void analyzeStageDefinitions(); 

	TupleDef *getEnvTuple() { return envTuple; }
        TupleDef *getPartitionTuple() { return partitionTuple; }
	List<Type*> *getInitArgTypes();
	int getPartitionArgsCount();

	// The custom types for task's define, environment, and partition sections are created before a 
	// full scale type checking of the task.
	void attachScope(Scope *parentScope);

	// The Initialize Section, if exists, should be validated first so that the arguments of any task 
	// invocation found in the coordinator function can be validated.  
	void typeCheckInitializeSection(Scope *scope);

	// This interpretes the instructions of the Partition Section, validates those instructions, and 
	// constructs a hierarchical description of LPSes and their data content.
	void constructPartitionHierarchy();

	// This method generates a computation flow for the task by combining information from the Stages
	// and Computation Sections and also validates that flow during the construction process.
	void constructComputationFlow(Scope *programScope);

	// the final round of semantic validation of the task is done here
	void validateScope(Scope *parentScope);

	 //-------------------------------------------------------------------- Helper functions for Static Analysis

	// this function embodies all static analyses relevent to the context of a single task execution
        void performStaticAnalysis();

	// This function returns information regarding the nature of access of all environmental data 
        // structures of the task. It must be called after variable access analysis is done for individual
        // compute stages. This function just accumulates information of selective data structures from the 
        // statistics generated by the earlier analyses. 
        List<VariableAccess*> *getAccessLogOfEnvVariables();

	// This function returns a structure retaining both the read/update information of environmental
        // data structures and the stale/fresh status of their different LPS allocations at the end of the 
        // task execution. 
        TaskEnvStat *getAfterExecutionEnvStat();

  public:
	// a static reference to the current task to facilitate access during different sub-phases of 
	// semantic and static analyses
	static TaskDef *currentTask;
	 
	//------------------------------------------------------------- Common helper functions for Code Generation

	CompositeStage *getComputation(); 
	InitializeSection *getInitSection() { return initialize; }
	List<EnvironmentLink*> *getEnvironmentLinks();
	PartitionHierarchy *getPartitionHierarchy();
	List<const char*> *getRepeatIndexes();
	const char *getName() { return id->getName(); }
	List<Identifier*> *getPartitionArguments();

	// As the name suggests, this function retrieves the header files to be included during compilation
        // and libraries to be linked during code generation for successful execution of all extern code 
	// blocks used within an IT task.
        IncludesAndLinksMap *getExternBlocksHeadersAndLibraries();
};

#endif
