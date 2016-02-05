#ifndef _H_data_flow
#define _H_data_flow

#include "ast.h"
#include "ast_expr.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "task_space.h"
#include "hashtable.h"

class VariableAccess;
class DataDependencies;

/*	This version of Data Flow Stage is no longer in use. We keep it around for now as it has some utility routines
	that are used for creating the flow stages listed below.
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

        void setNestingIndex(int nestingIndex) { this->nestingIndex = nestingIndex; }
	void setNestingController(DataFlowStage *controller) { this->nestingController = controller; }
	int getNestingIndex() { return nestingIndex; }
        void setComputeIndex(int computeIndex) { this->computeIndex = computeIndex; }
	int getComputeIndex() { return computeIndex; }
	DataDependencies *getDataDependencies();
        virtual Hashtable<VariableAccess*> *getAccessMap() { return accessMap; }
        Space *getSpace() { return executionSpace; }
	const char *performDependencyAnalysis(List<DataFlowStage*> *stageList);
};

/* 	Task global variables may be synchronized/retrieved in several cases. The cases are
	Entrance: moving from a higher to a lower space
	Exit: exiting from a lower to higher space
	Return: returning from a lower to a higher space
	Reappearance: moving from one stage of a space to another of the same space
*/
enum SyncStageType { Entrance, Exit, Return, Reappearance };

/*	Depending on the type of sync stage synchronization need is different. Scenarios are
	Load: need to read some data structures into a space
	Load_And_Configure: need to read data structures and also to update metadata of those structures
	Ghost_Region_Update: need to do padding region synchronization
	Restore: Need to upload changes from below to upper region for persistence
*/
enum SyncMode { Load, Load_And_Configure, Ghost_Region_Update, Restore };

/*	Repeat Cycles can be of two types
	Subpartition_Repeat: repeat one or more execution stage for all sub-partitions of sub-partitioned
			     variables of a space. Not that variables that are not sub-partitioned will 
			     be loaded once.
	Conditional_Repeat: is a generic loop condition based repeat of execution stages
*/
enum RepeatCycleType { Subpartition_Repeat, Conditional_Repeat };


/*	Base class for representing a stage in the execution flow of a task. Instead of directly using
	the compute and meta-compute stages that we get from the abstract syntax tree, we derive a modified
	set of flow stages that are easier to reason with for later part of the compiler. 
*/
class FlowStage {
  protected:
	int index;
	const char *name;
	Space *space;
	Hashtable<VariableAccess*> *accessMap;
	Expr *executeCond;  
        DataDependencies *dataDependencies;
  public:
	FlowStage(int index, Space *space, Expr *executeCond);
	virtual ~FlowStage() {};
	void setIndex(int index) { this->index = index; }
	int getIndex() { return index; }
	const char *getName() { return name; }
	void setName(const char *name) { this->name = name; }
	Space *getSpace() { return space; }
	Hashtable<VariableAccess*> *getAccessMap() { return accessMap; }
	void setAccessMap(Hashtable<VariableAccess*> *accessMap) { this->accessMap = accessMap; }
	void mergeAccessMapTo(Hashtable<VariableAccess*> *destinationMap);
	void addAccessInfo(VariableAccess *accessLog);
	Hashtable<VariableAccess*> *getAccessLogsForSpaceInIndexLimit(Space *space, 
			List<FlowStage*> *stageList, int startIndex, int endIndex, 
			bool includeMentionedSpace);
	Hashtable<VariableAccess*> *getAccessLogsForReturnToSpace(Space *space,
			List<FlowStage*> *stageList, int endIndex);
	virtual void print(int indent);
	virtual void performDependencyAnalysis(PartitionHierarchy *hierarchy) { 
		performDependencyAnalysis(accessMap, hierarchy); 
	}
	void performDependencyAnalysis(Hashtable<VariableAccess*> *accessLogs, PartitionHierarchy *hierarchy);
	DataDependencies *getDataDependencies() { return dataDependencies; }
	bool isDataModifierRelevant(FlowStage *modifier);

	// This virtual method is added so that loader sync stages for dynamic spaces can be put inside composite 
	// stages protected by activation conditions alongside their compute stages. The normal flow construction
	// algorithm keeps them outside. Which is semantically equivalent to allocate and load data for all LPUs
	// but execute only those with activation condition evaluated to true. That defeats the advantage of having
	// dynamic stages altogether.	 
	virtual void reorganizeDynamicStages() {}
	void setExecuteCondition(Expr *condition) { executeCond = condition; }
	Expr *getExecuteCondition() { return executeCond; }
};

/*	Sync stages are automatically added to the user specified execution flow graph during static analysis.
	These stages have no code within. They only keep track of the data structures need to be synchronized. 
*/
class SyncStage : public FlowStage {
  protected:
	SyncMode mode;
	SyncStageType type;
  public:
	SyncStage(Space *space, SyncMode mode, SyncStageType type);
	int populateAccessMap(List<VariableAccess*> *accessLogs, 
		bool filterOutNonReads, bool filterOutNonWritten);
	bool isLoaderSync() { return (mode == Load || mode == Load_And_Configure); }
};

/*	This class is equivalent to a ComputeStage with executable code from abstract syntax tree
*/
class ExecutionStage : public FlowStage {
  protected:
	Stmt *code;
  public:
	ExecutionStage(int index, Space *space, Expr *executeCond);
	void setCode(List<Stmt*> *stmtList);
};

/*	Composite stage construct is similar to a meta compute stage of the abstract syntax tree. It is much 
	simplified though. We can think it as simple linear sequence of stages without any repeat cycle. So 
	unless nested in a repeat cycle, a composite Stage will execute only once. 
*/
class CompositeStage : public FlowStage {
  protected:
	List<FlowStage*> *stageList;
  public:
	CompositeStage(int index, Space *space, Expr *executeCond);
	virtual ~CompositeStage() {}
	void addStageAtBeginning(FlowStage *stage);
	void addStageAtEnd(FlowStage *stage);
	void insertStageAt(int index, FlowStage *stage);
	void removeStageAt(int stageIndex);
	bool isStageListEmpty();
	Space *getLastNonSyncStagesSpace();
	FlowStage *getLastNonSyncStage();
	void setStageList(List<FlowStage*> *stageList) { this->stageList = stageList; }
	List<FlowStage*> *getStageList() { return stageList; }
	void addSyncStagesBeforeExecution(FlowStage *nextStage, List<FlowStage*> *stageList);
	virtual void addSyncStagesOnReturn(List<FlowStage*> *stageList);
	virtual void print(int indent);
	virtual void performDependencyAnalysis(PartitionHierarchy *hierarchy);
	void reorganizeDynamicStages();
};

/*	A repeat cycle is a composite stage iterated one or more times under the control of a repeat instruction.
*/
class RepeatCycle : public CompositeStage {
  protected:
	RepeatCycleType type;
	Expr *repeatCond;
	Hashtable<VariableAccess*> *repeatConditionAccessMap;
  public:
	RepeatCycle(int index, Space *space, RepeatCycleType type, Expr *executeCond);
	void addSyncStagesOnReturn(List<FlowStage*> *stageList);
	void setRepeatConditionAccessMap(Hashtable<VariableAccess*> *map) { repeatConditionAccessMap = map; }
	void performDependencyAnalysis(PartitionHierarchy *hierarchy);
};

/*	This is a utility class to keep track of the last point of entry to a space as flow of control move from
	flow stages to flow stages. This is required so that during a space exit we can determine what previous 
	stages to check for potential data structure changes that we may need to synchronize. A reference to any
	possible sync stage associated with space entry is also maintained to be updated at space exit as to read
	only those data structures that are been accessed in actual computation in between entry and exit. 
*/
class SpaceEntryCheckpoint {
  protected:
	Space *space;
	int entryStageIndex;
	SyncStage *entrySyncStage;
  public:
	static SpaceEntryCheckpoint *addACheckpointIfApplicable(Space *space, int stageIndex);
	static void removeACheckpoint(Space *space);
	static SpaceEntryCheckpoint *getCheckpoint(Space *space);
	int getStageIndex() { return entryStageIndex; }
	void setEntrySyncStage(SyncStage *entrySyncStage) { this->entrySyncStage = entrySyncStage; }
	SyncStage *getEntrySyncStage() { return entrySyncStage; }		
  private:	
	SpaceEntryCheckpoint(Space *space , int entryStageIndex);
	static Hashtable<SpaceEntryCheckpoint*> *checkpointList;
};

/*	This is a utility class that checks space transitions as the flow of control moves from flow stages to
	stages, and determine what type of synchronization stages should be put in-between.	
*/
class SyncStageGenerator {
  public:
	static bool doesTransitionNeedSynchronization(Space *previousSpace, Space *nextSpace);
	static SyncStage *generateEntrySyncStage(Space *space);
	static void populateAccessMapOfEntrySyncStage(SyncStage *stage, Hashtable<VariableAccess*> *accessLogs);
	static SyncStage *generateReappearanceSyncStage(Space *space, Hashtable<VariableAccess*> *accessLogs);
	static SyncStage *generateReturnSyncStage(Space *space, Hashtable<VariableAccess*> *accessLogs);
	static List<SyncStage*> *generateExitSyncStages(Space *space, Hashtable<VariableAccess*> *accessLogs);  
	static List<VariableAccess*> *generateListFromLogs(Hashtable<VariableAccess*> *accessLogs);
	static List<VariableAccess*> *filterAccessList(List<VariableAccess*> *accessList, 
			List<const char*> *includeIfExistsList);	
};

/*	This is merely an utility class to keep track what data structure should be synchronized to what
	ancestor space when flow of control exits from a space whose data structures wont last after the
	exit.
*/
class ExitSpaceToDataStructureMappings {
  protected:
	Space *ancestorSpace;
	List<VariableAccess*> *accessList;
  public:
	ExitSpaceToDataStructureMappings(Space *ancestorSpace);
	void generateAccessInfo(const char *varName);
	bool isAccessListEmpty() { return (accessList->NumElements() == 0); }
	SyncStage *generateSyncStage();		
};

#endif
