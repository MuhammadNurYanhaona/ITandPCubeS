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
};

/*	A stage instanciation represents an invocation done from the Computation Section of a compute stage defined 
	in the Stages Section
*/
class StageInstanciation : public FlowStage {
  protected:
	Stmt *code;
	Scope *scope;
	const char *name;
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

	void addStageAtBeginning(FlowStage *stage);
	void addStageAtEnd(FlowStage *stage);
	void insertStageAt(int index, FlowStage *stage);
	void removeStageAt(int stageIndex);

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
};

/*	A repeat control block is a composite stage being iterated over under the control of a repeat instruction.
*/
class RepeatControlBlock : public CompositeStage {
  protected:
	Expr *condition;
	RepeatCycleType type;
  public:
	RepeatControlBlock(Space *space, RepeatCycleType type, Expr *executeCond);
	void print(int indent);
	void performDataAccessChecking(Scope *taskScope);
	
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

#endif
