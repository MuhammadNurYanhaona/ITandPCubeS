#ifndef _H_computation_flow
#define _H_computation_flow

#include "scope.h"
#include "task_space.h"
#include "../common/location.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

class VariableAccess;
class CompositeStage;
class ReductionMetadata;

/*	Base class for representing a stage in the execution flow of a task. Instead of directly using the compute and 
	meta-compute stages that we get from the abstract syntax tree, we derive a modified set of flow stages that are 
	easier to reason with for later part of the compiler. 
*/
class FlowStage {
  protected:
	Space *space;
	FlowStage *parent;
	
	// Index indicates the position of a flow stage compared to other stages and group No specifies its container 
        // stage's, if exists, index. Finally, repeat index is the index of the closest repeat control block that 
	// encircles current stage. These positional properties are needed for different kinds of data dependency and
	// access analyses.
        int index;
        int groupNo;
        int repeatIndex;

	// an assigned location information for the flow stage to be used for error reporting purpose
	yyltype *location;

	// a map that tracks use of task-global variables in the current flow stage
	Hashtable<VariableAccess*> *accessMap;

	// This list stores data structures used in epoch expressions inside the subflow/computation represented by 
        // this flow stage. This information is later used to advance appropriate data structures' epoch version after
        // the execution of the flow stage.
        List<const char*> *epochDependentVarList;	
  public:
	FlowStage(Space *space);
	virtual ~FlowStage() {};
	void setParent(FlowStage *parent) { this->parent = parent; }
	FlowStage *getParent() { return parent; }
	Space *getSpace() { return space; }
	Hashtable<VariableAccess*> *getAccessMap();
	void assignLocation(yyltype *location) { this->location = location; }
	yyltype *getLocation() { return location; }
	virtual void print(int indent) = 0;
	
	void setIndex(int index) { this->index = index; }
	int getIndex() { return index; }
	void setGroupNo(int index) { this->groupNo = index; }
	int getGroupNo() { return groupNo; }
	void setRepeatIndex(int index) { this->repeatIndex = index; }
	int getRepeatIndex() { return repeatIndex; }

	// this is the interface for a recursive routine that investigate the use of task-global variables in the 
	// computation flow 
	virtual void performDataAccessChecking(Scope *taskScope) = 0;
  protected:
	// an utility function to be used during data access analysis of flow stages to ensure that access to any task
	// global variable done from a flow stage is permitted in the LPS the stage is going to execute
	// after validation it also produces an access-map from the activation condition and the code in case storing
	// the access-map might be usefull
	Hashtable<VariableAccess*> *validateDataAccess(Scope *taskScope, Expr *activationCond, Stmt *code);
  public:
	//------------------------------------------------------------------------ Helper functions for Static Analysis

	// utility functions needed for various static analyses--------------------------------------------------------

	// This function returns the top-most LPS that is neither the container stage's LPS nor the contained stage's
	// LPS. This utility is needed to detect LPS crossing transitions in the computation flow. If there is no LPS
	// crossing among the argument stages then it returns NULL.
	static Space *getCommonIntermediateLps(FlowStage *container, FlowStage *contained);

	// A recursive process to assign index and group no to flow stages
        virtual int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);

	//-------------------------------------------------------------------------------------------------------------
	
	// functions related to sync stage implantation in the compute flow--------------------------------------------
	
	// When the partition hierarchy has LPSes having sub-partitions, overlapping data structure partitions, etc.
	// then the compiler needs to implant sync-stages after execution of stages in such LPSes. This is the first
	// step of static analysis. This function does the implantation using a recursive process. 
	virtual void implantSyncStagesInFlow(CompositeStage *containerStage, List<FlowStage*> *currStageList);

	// this function is needed to retrieve information about task-global variable accesses done from a given LPS
	// or from its descendent LPSes within the current flow-stage
	virtual void populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress, 
			Space *lps, bool includeLimiterLps) = 0;
  protected:	
	// two utility functions needed for augmenting sync-stages in the computation flow
	Hashtable<VariableAccess*> *getAccessLogsForSpaceInIndexLimit(Space *space, 
			List<FlowStage*> *stageList, 
			int startIndex, 
			int endIndex, bool includeMentionedSpace);
	Hashtable<VariableAccess*> *getAccessLogsForReturnToSpace(Space *space, 
			List<FlowStage*> *stageList, int endIndex);
	
	//-------------------------------------------------------------------------------------------------------------

  public:
	// functions for annotating LPSes and flow stages about data structure usage statistics------------------------
	
	// This routine mainly serves the purpose of annotating a task with information about different data structure 
	// usage in different LPSes. This knowledge is important regarding data structure generation and memory 
	// allocation for variables in any backend; 
        virtual void calculateLPSUsageStatistics();

	// A recursive process to determine how many versions of different data structures need to be maintained for
        // epoch dependent calculations happening within the flow stages.
        virtual void performEpochUsageAnalysis() {}
	// a static reference to the current flow stage to be accessible from statements and expressions for any 
	// recursive analysis of code the flow stage contains
	static FlowStage *CurrentFlowStage;
	List<const char*> *getEpochDependentVarList() { return epochDependentVarList; }

	// an analysis to flag LPSes to indicate some computation occurs within it; this is needed to determine whether 
	// or not to generate LPUs for an LPS
        virtual void setLpsExecutionFlags() {}
	// this function tells if any runtime logic related to the current flow stage requires accessing LPU data
	bool isLpsDependent();

	//-------------------------------------------------------------------------------------------------------------

	// functions for task environment processing and analysis------------------------------------------------------
	
	// We need to determine what environmental variables are read-only and what are read-write within a task
        // to determine what impact a task's execution should have on the overall program environment. This method
        // recursively fills a list of empty environmental variable access objects with actual access information
        // to aid in the program environment management process.
        virtual void fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList);

        // The functionality of this method overlaps with the previous fillInTaskEnvAccessList() method. It tracks
        // the usage of environmental variables within the task. It also tracks the states different memory allocations 
	// for a single data structure at the end of the execution of the computation flow in a task environment
        // statistic variable. For the first part, it uses the result of the previous function.  
        virtual void prepareTaskEnvStat(TaskEnvStat *taskStat);

	//-------------------------------------------------------------------------------------------------------------

	// functions for flow expansion to incorporate reductions------------------------------------------------------

	// These two auxiliary functions are needed to all process parallel reductions found in the computation
        // flow of a task. The first function 'populateReductionMetadata' investigates the statements of Execution
        // stages and discover any reductions in them. The second function 'extractAllReductionInfo' lists all 
        // reductions of a task so that proper memory management decisions can be made regarding the reduction result 
        // variables. 
        virtual void populateReductionMetadata(PartitionHierarchy *lpsHierarchy) {}
        virtual void extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos) = 0;
	
	// The logic of encoding reduction management instructions in the compute flow is that reduction instructions 
	// are uplifted from nested compute stages to appropriate LPS level. Then reduction boundary flow stages are 
	// inserted that willtake care of any resource setup/teardown and data exchange needed during the code 
	// generation step. This function does the uplifting. 
	virtual List<ReductionMetadata*> *upliftReductionInstrs() = 0;

	// This utility function filters out all those reductions from the flow stage and stages nested within it 
	// that has the argument LPS as the reduction root. The filtered reductions' metadata is added to the second
	// argument of the function. 
	virtual void filterReductionsAtLps(Space *reductionRootLps, List<ReductionMetadata*> *filteredList) = 0;

	// Variables holding the result of reduction cannot be used until all LPU computations contributing to the
	// reductions are done. These two functions are needed to check if there is any incorrect use of a reduction
	// result before it is ready.  
	virtual FlowStage *getLastAccessorStage(const char *varName) { return NULL; }
	virtual void validateReductions() {}

	//-------------------------------------------------------------------------------------------------------------
};

/*	A stage instanciation represents an invocation done from the Computation Section of a compute stage defined 
	in the Stages Section
*/
class StageInstanciation : public FlowStage {
  protected:
	Stmt *code;
	Scope *scope;
	const char *name;

	// this holds metadata about all reduction statements found within this compute stage instance
        List<ReductionMetadata*> *nestedReductions;
  public:
	StageInstanciation(Space *space);
	void setCode(Stmt *code) { this->code = code; }
	void setCode(List<Stmt*> *stmtList);
	void setScope(Scope *scope) { this->scope = scope; }
	void setName(const char *name) { this->name = name; }
	const char *getName() { return name; }
	Scope *getScope() { return scope; }
	void print(int indent);
	void performDataAccessChecking(Scope *taskScope);
	
	//------------------------------------------------------------------------ Helper functions for Static Analysis
	
	// functions related to sync stage implantation in the compute flow--------------------------------------------
	
	void populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress, 
			Space *lps, bool includeLimiterLps);
	
	//-------------------------------------------------------------------------------------------------------------
	
	// functions for annotating LPSes and flow stages about data structure usage statistics------------------------
        
	void performEpochUsageAnalysis();
	void setLpsExecutionFlags();

	//-------------------------------------------------------------------------------------------------------------
	
	// functions for flow expansion to incorporate reductions------------------------------------------------------
	
	void populateReductionMetadata(PartitionHierarchy *lpsHierarchy);
        void extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos);
	List<ReductionMetadata*> *upliftReductionInstrs();
	void filterReductionsAtLps(Space *reductionRootLps, List<ReductionMetadata*> *filteredList);
	FlowStage *getLastAccessorStage(const char *varName);

	//-------------------------------------------------------------------------------------------------------------
};

/*	A composite stage is a holder of other flow stages and control blocks as a sub-flow. */
class CompositeStage : public FlowStage {
  protected:
	List<FlowStage*> *stageList;
  public:
	CompositeStage(Space *space);
	virtual ~CompositeStage() {}
	void setStageList(List<FlowStage*> *stageList);
	List<FlowStage*> *getStageList() { return stageList; }
	virtual void print(int indent);
	virtual void performDataAccessChecking(Scope *taskScope);

	//------------------------------------------------------------------------ Helper functions for Static Analysis

	// utility functions needed for various static analyses--------------------------------------------------------
	
	void addStageAtBeginning(FlowStage *stage);
	void addStageAtEnd(FlowStage *stage);
	void insertStageAt(int index, FlowStage *stage);
	void removeStageAt(int stageIndex);
        virtual int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);
	
	// This recursive function modifies the internal organization of flow stages inside a composite stage by 
        // making all LPS transitions from container stages to the contained stages explicit. To give an example, if
        // LPS hierarchy is like Space C divides B divides A and a composite stage at Space A has a nested stage at
        // Space C, this adjustment will surround the nested stage within another Space B composite stage. Explicit
        // LPS transitions like the aforementioned example are important for several code generation reasons.
        void makeAllLpsTransitionsExplicit();

	//-------------------------------------------------------------------------------------------------------------
	
	// functions related to sync stage implantation in the compute flow--------------------------------------------
	
	virtual void implantSyncStagesInFlow(CompositeStage *containerStage, List<FlowStage*> *currStageList);
	void populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress, 
			Space *lps, bool includeLimiterLps);
	bool isStageListEmpty();

	// swaps the current flow-stage list with the argument flow-stage list and returns the old list
	List<FlowStage*> *swapStageList(List<FlowStage*> *argList);

	// thess two functions are needed for incremental re-construction of the flow-stage list of the current 
	// composite stage 
	Space *getLastNonSyncStagesSpace();
	FlowStage *getLastNonSyncStage();

	// these two functions embodies the logic of sync-stage implantation
	void addSyncStagesBeforeExecution(FlowStage *nextStage, List<FlowStage*> *stageList);
	void addSyncStagesOnReturn(List<FlowStage*> *stageList);
	
	//-------------------------------------------------------------------------------------------------------------
	
	// functions for annotating LPSes and flow stages about data structure usage statistics------------------------
        
	virtual void calculateLPSUsageStatistics();
        virtual void performEpochUsageAnalysis();
	virtual void setLpsExecutionFlags();

	//-------------------------------------------------------------------------------------------------------------
	
	// functions for task environment processing and analysis------------------------------------------------------
        
	virtual void fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList);
        virtual void prepareTaskEnvStat(TaskEnvStat *taskStat);
	
	//-------------------------------------------------------------------------------------------------------------
	
	// functions for flow expansion to incorporate reductions------------------------------------------------------

	virtual void populateReductionMetadata(PartitionHierarchy *lpsHierarchy);
        virtual void extractAllReductionInfo(List<ReductionMetadata*> *reductionInfos);	
	virtual List<ReductionMetadata*> *upliftReductionInstrs();
	virtual void filterReductionsAtLps(Space *reductionRootLps, List<ReductionMetadata*> *filteredList);
	virtual FlowStage *getLastAccessorStage(const char *varName);
	virtual void validateReductions();

	//-------------------------------------------------------------------------------------------------------------
};

/*	A repeat control block is a composite stage being iterated over under the control of a repeat instruction.
*/
class RepeatControlBlock : public CompositeStage {
  protected:
	Expr *condition;
	RepeatCycleType type;
  public:
	RepeatControlBlock(Space *space, RepeatCycleType type, Expr *executeCond);
	void performDataAccessChecking(Scope *taskScope);
	void print(int indent);
	
	//------------------------------------------------------------------------ Helper functions for Static Analysis
	
	// utility functions needed for various static analyses--------------------------------------------------------
        
	int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);
	
	//-------------------------------------------------------------------------------------------------------------
	
	// functions for annotating LPSes and flow stages about data structure usage statistics------------------------
        
	void calculateLPSUsageStatistics();
        void performEpochUsageAnalysis();
	void setLpsExecutionFlags();

	//-------------------------------------------------------------------------------------------------------------
	
	// functions for task environment processing and analysis------------------------------------------------------
        
	virtual void fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList);
        virtual void prepareTaskEnvStat(TaskEnvStat *taskStat);
	
	//-------------------------------------------------------------------------------------------------------------
	
	// functions for flow expansion to incorporate reductions------------------------------------------------------
	
	List<ReductionMetadata*> *upliftReductionInstrs();

	//-------------------------------------------------------------------------------------------------------------
};

/*	A conditional execution block represents a composite stage that has the nested sub-flow set to be executed
	only if a particular condition is true. 
*/
class ConditionalExecutionBlock : public CompositeStage {
  protected:
	Expr *condition;
  public:
	ConditionalExecutionBlock(Space *space, Expr *executeCond);
	void print(int indent);
	void performDataAccessChecking(Scope *taskScope);
	
	//------------------------------------------------------------------------ Helper functions for Static Analysis
	
	// functions for annotating LPSes and flow stages about data structure usage statistics------------------------
        
	void calculateLPSUsageStatistics();
        void performEpochUsageAnalysis();
	void setLpsExecutionFlags();

	//-------------------------------------------------------------------------------------------------------------
	
	// functions for task environment processing and analysis------------------------------------------------------
        
	virtual void fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList);
        virtual void prepareTaskEnvStat(TaskEnvStat *taskStat);
	
	//-------------------------------------------------------------------------------------------------------------
};

/*	This represents a transition in the Computation flow of the task from an ancestor LPS to a descendent LPS.
*/
class LpsTransitionBlock : public CompositeStage {
  protected:
	Space *ancestorSpace;
  public:
	LpsTransitionBlock(Space *space, Space *ancestorSpace);		
	void print(int indent);
};

/*	This represents a sub-flow boundary at the end of which the versions of all multi-version data structures
	used within the sub-flow must be advanced by one step.
*/
class EpochBoundaryBlock : public CompositeStage {
  public:
	EpochBoundaryBlock(Space *space);	
	void print(int indent);
};

/*	This represents a compiler generated code-block boundary added for setup and tear down of resources related
	reductions found in the nested stages.
*/
class ReductionBoundaryBlock : public CompositeStage {
  protected:
	// this list holds information about all reduction operations this stage is responsible for
	List<ReductionMetadata*> *assignedReductions;
  public:
	ReductionBoundaryBlock(Space *space);
	void print(int indent);
	
	//------------------------------------------------------------------------ Helper functions for Static Analysis
	
	// functions for flow expansion to incorporate reductions------------------------------------------------------
	
	void assignReductions(List<ReductionMetadata*> *reductionList);
	void validateReductions();

	//-------------------------------------------------------------------------------------------------------------
};

#endif
