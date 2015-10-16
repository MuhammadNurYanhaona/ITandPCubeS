#ifndef _H_data_flow
#define _H_data_flow

#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"
#include "../semantics/scope.h"
#include "../utils/hashtable.h"

#include <iostream>
#include <fstream>

class VariableAccess;
class DataDependencies;
class StageSyncReqs;
class StageSyncDependencies;
class SyncRequirement;
class CommunicationCharacteristics;

/* An important point to remember about synchronization related enums and classes in this headers is that they corresponds 
   to situations where there is a need for definite data movements along with signaling the fact that an update has taken 
   place. The information regarding the signaling requirement is retained in the dependency arcs and some other variables 
   that are kept in each flow stages. Forgetting this issue may produce confusions in the understanding of the intended 
   implementation of synchronization stages. To clarify this distinction with an example, Entrance, return and Exit syncs 
   are put in place only for dynamic and subpartitioned LPSes. Reappearance syncs are only for LPSes having overlapping 
   ghost regions in some data structure partitions.      
*/

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

/*	Base class for representing a stage in the execution flow of a task. Instead of directly using the compute and 
	meta-compute stages that we get from the abstract syntax tree, we derive a modified set of flow stages that are 
	easier to reason with for later part of the compiler. 
*/
class FlowStage {
  protected:
	const char *name;
	Space *space;
	Hashtable<VariableAccess*> *accessMap;
	Expr *executeCond;  
	
	// Index indicates the position of a flow stage compared to other stages and group No specifies its container 
	// stage's, if exists, index. Finally, repeat index is the index of the closest repeat  cycle then encircles 
	// current stage. 
	int index;
	int groupNo;
	int repeatIndex;
        
	// three data structures that track all communication and synchronization requirements related to a flow stage
	DataDependencies *dataDependencies;
	StageSyncReqs *synchronizationReqs;
	StageSyncDependencies *syncDependencies;
  public:
	FlowStage(int index, Space *space, Expr *executeCond);
	virtual ~FlowStage() {};
	void setIndex(int index) { this->index = index; }
	int getIndex() { return index; }
	void setGroupNo(int groupNo) { this->groupNo = groupNo; }
	int getGroupNo() { return groupNo; }
	void setRepeatIndex(int repeatIndex) { this->repeatIndex = repeatIndex; }
	int getRepeatIndex() { return repeatIndex; }
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
	
	// A recursive process to determine how many versions of different data structures need to be maintained for
	// epoch dependent calculatations happening within the flow stages.
	virtual void performEpochUsageAnalysis() {}

	DataDependencies *getDataDependencies() { return dataDependencies; }
	bool isDataModifierRelevant(FlowStage *modifier);

	// A recursive process to assign index and group no to flow stages; composite flow stage override this method
	// to implement recursion. For other stages default implementation works, which just assign the passed
	// arguments. 
	virtual int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);

	// This virtual method is added so that loader sync stages for dynamic spaces can be put inside composite 
	// stages protected by activation conditions alongside their compute stages. The normal flow construction
	// algorithm keeps them outside. Which is semantically equivalent to allocate and load data for all LPUs
	// but execute only those with activation condition evaluated to true. That defeats the advantage of having
	// dynamic stages altogether.	 
	virtual void reorganizeDynamicStages() {}
	void setExecuteCondition(Expr *condition) { executeCond = condition; }
	Expr *getExecuteCondition() { return executeCond; }

	// This virtual method is used to recursively generate the run method for the task for the multi-core back
	// end compiler.
	virtual void generateInvocationCode(std::ofstream &stream, 
			int indentation, Space *containerSpace) {}

	// For code generation for the back-end multi-core compiler we need to know if a flow stage should be enter-
	// ed by multiple PPUs that are under a single higher level PPU but not necessarily responsible for
	// executing any code of current LPS under concern; or only the PPU with valid PPU ID for the LPS should
	// enter the code. To make it clear with an example, if an execution stage has a reduction instruction then
	// all PPUs within a group should participate in that reduction. If it is a stage devoid of instructions 
	// needing such collective support, then only a single PPU should enter and execute it. The method below 
	// need to be overridden by subclasses to reflect intended behavior. 
	virtual bool isGroupEntry() { return false; }
	
	// This method is required to determine what variables need to be copied in local socpe from the LPU for
	// the flow stage to simplify code generation.
	List<const char*> *filterInArraysFromAccessMap(Hashtable<VariableAccess*> *accessMap = NULL);

	// This is a static analysis routine that mainly serves the purpose of annotating a task with information 
	// about different data structure usage in different LPSes. This knowledge is important regarding data 
	// structure generation and memory allocation for variables in any backend; 
	virtual void calculateLPSUsageStatistics();

	// This analysis examines the dependency arcs associated with this flow stage and determines what should be
	// encoded as a synchronization requirement and what should results in mere ordering of flow stages during
	// code execution. 
	virtual void analyzeSynchronizationNeeds();
	
	// Calling these routine make sense only ofter the synchronization requirement analyis is done
	StageSyncReqs *getAllSyncRequirements();
	StageSyncDependencies *getAllSyncDependencies();
	virtual void printSyncRequirements(int indentLevel);

	// this function indicates if there is any synchronization requirement between the execution of the current
	// and the execution of the stage passed as argument
	bool isDependentStage(FlowStage *suspectedDependent);

	// this is a helper method for code generations; it helps in determining where to declare the counter
	// variables (that is used to determine how many times the updater of a to-be-synchronized data structure
	// executes)
	virtual List<const char*> *getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel);

	// this function is used to determine what reader of a data modified by the current stage should notify it
	// about completion of read so that this stage can execute again if need may be.
	virtual void setReactivatorFlagsForSyncReqs();

	// an analysis to flag LPSes to indicate some computation has taken place within it; this is needed to 
	// determine whether or not to generate LPUs for an LPS
	virtual void setLpsExecutionFlags() {}

	// a recursive routine to retrieve all data dependency relationships within a task; this is used for a later
	// analysis during code generation regarding what data to exchange among PPU controllers; obviously, this is 
	// only usable after dependency analysis has been done. 
	virtual List<DependencyArc*> *getAllTaskDependencies();

	// This function is used to recursively retrieves all variables used in the task that will be communicated 
	// across segments or within different allocations within a segment as part of some synchronization process.
	// Notice that it needs the Id of the physical Space (PPS) where memory segmentation has taken place. Thus
	// using it is only meaningful after LPS-to-PPS mappings are known  
	virtual List<const char*> *getVariablesNeedingCommunication(int segmentedPPS);
	// This function uses the same recursive process, but it returns detail communication information 
	virtual List<CommunicationCharacteristics*> *getCommCharacteristicsForSyncReqs(int segmentedPPS);
};

/*	Sync stages are automatically added to the user specified execution flow graph during static analysis. These 
	stages have no code within. They only keep track of the data structures need to be synchronized. 
*/
class SyncStage : public FlowStage {
  protected:
	SyncMode mode;
	SyncStageType type;

	// A sync stage exists to support data read/write and synchronization for compute stages. Although it has 
	// an accessed-variables list, that we assign to it to draw the data dependency arcs properly. When comes 
	// the implementation of data load/store/synchronization as dictated by those dependency arcs, we need to
	// find out the actual execution stages that did the data write. Therefore, a map of previous data modifier 
	// flow stages is maintained in sync-stages to track down the actual modifiers.  
	Hashtable<FlowStage*> *prevDataModifiers;
  public:
	SyncStage(Space *space, SyncMode mode, SyncStageType type);
	int populateAccessMap(List<VariableAccess*> *accessLogs, 
		bool filterOutNonReads, bool filterOutNonWritten);
	void performDependencyAnalysis(PartitionHierarchy *hierarchy); 
	bool isLoaderSync() { return (mode == Load || mode == Load_And_Configure); }
	void analyzeSynchronizationNeeds();
	void generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace);
	
	// For now usage statistics is not been gathered for sync stages 
	void calculateLPSUsageStatistics() {}

	// returns the execution stage that did actual write of a structure listed as modified my this sync stage
	FlowStage *getUltimateModifier(const char *varName);
};

/*	This class is equivalent to a ComputeStage with executable code from abstract syntax tree
*/
class ExecutionStage : public FlowStage {
  protected:
	Stmt *code;
	Scope *scope;
  public:
	ExecutionStage(int index, Space *space, Expr *executeCond);
	void setCode(List<Stmt*> *stmtList);
	void setScope(Scope *scope) { this->scope = scope; }
	Scope *getScope() { return scope; }
	void performEpochUsageAnalysis();

	// helper method for generating back-end code
	void translateCode(std::ofstream &stream);
	void generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace);
	bool isGroupEntry();
	void setLpsExecutionFlags();
};

/*	Composite stage construct is similar to a meta compute stage of the abstract syntax tree. It is much 
	simplified though. We can think it as simple linear sequence of stages without any repeat cycle. So unless 
	nested in a repeat cycle, a composite Stage will execute only once.

	NOTE NOTE NOTE NOTE: data dependencies in an IT task are distinguished between synchronization dependencies
	and communication dependencies based on the target hardware platfrom and how LPS-to-PPS mapping is done. The
	strategy we applied to drag LPU-LPU dependencies to PPU-PPU (note that a segment is a higher level PPU by 
	itself) moves all data dependencies from their execution and sync-stages to encompassing composite stages.
	Thus, here we separate the two types of dependencies and generate different code for them. For communication
	dependencies we just need a send and a receive. For synchronization dependencies, in its most flexible mode,
	there should be a signalUpdate-waitForUpdate-signalRead-waitForRead sync cycle giving four functions. The
	implementation of the underlying synchronization primitives (done by Profe.), however, tied the signal with
	wait like a barrier fashion. So although we originally had template for four synchronization functions. We
	did not use them later. Rather we have two simplified functions for signalUpdate-waitForRead shorter cycle.
	The bottom line is do not get frightened by two many data dependency resolution functions here.       
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
	virtual void performEpochUsageAnalysis();
	void reorganizeDynamicStages();
	virtual void calculateLPSUsageStatistics();
	void analyzeSynchronizationNeeds();
	
	// composite stages do not have any synchronization dependencies of their own rather; they derive depend-
	// encies from stages nested within them. The analyze-synchronization-needs routines is not enough for
	// composite stages as all synchronization dependencies of nested stages are not resolved and available
	// for their composite stages to be derived from when the control of recursion is on the composite stages.
	// (All dependencies are resolved only after the recursion ends.) So this additional recursion is done
	// for composite stages after previous analysis.
	void deriveSynchronizationDependencies();

	// A composite stage by itself does not create any synchronization need. Rather it derives such needs
	// from stages embedded within it. Here the logic is that if the update within a nested stage creates a
	// dependency on some stage outside the composite stage boundary then it should be assigned to the 
	// composite stage. Recursive applying this logic ensures that all synchronization needs fall in the
 	// composite stage nesting in such a way that the updater and the receiver of that update are always 
	// within the same composite stage. Before this recursive procedure is done, we need to ensure that all
	// synchronization needs of nested stages along with their synchronization dependencies are set properly.
	// So this is the last method to envoke in the process of resolving synchronization.
	void analyzeSynchronizationNeedsForComposites();
	
	void printSyncRequirements(int indentLevel);
	virtual int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);

	// helper functions for code generation-------------------------------------------------------------------
	
	// A composite stage organize the stages within into groups based on their LPSes so that iterations over 
	// LPUs happens in group basis instead of individual computation basis. This reduces the number of times
	// we need to invoke the LPU generation library at runtime.
	List<List<FlowStage*>*> *getConsecutiveNonLPSCrossingStages();
	
	// override for the code generation method inherited from Flow-Stage class
	virtual void generateInvocationCode(std::ofstream &stream, 
			int indentation, Space *containerSpace);
	
	// Until we incorporate the backend support for dynamic LPSes, there is no need to generate codes for sync 
	// stages; therefore, we filter them out during nested stages grouping.
	static List<FlowStage*> *filterOutSyncStages(List<FlowStage*> *originalList);
	
	// For both multi-core and segmented-memeory backends, our current decision is to drag down LPU-LPU synch-
	// ronization to PPU-PPU synchronization. As we group flow-stages during code generations and iterations 
	// over multiplexed LPUs take place on a group basis, we need to know all data dependencies the stages of a 
	// group has and resolve them before we start code generation for the stages. The following three functions 
	// help in doing that.
	static List<SyncRequirement*> *getDataDependeciesOfGroup(List<FlowStage*> *group);
	void generateDataReceivesForGroup(std::ofstream &stream, int indentation, 
			List<SyncRequirement*> *commDependencies);
	void generateSyncCodeForGroupTransitions(std::ofstream &stream, int indentation, 
			List<SyncRequirement*> *syncDependencies);

	bool isGroupEntry();
	void setLpsExecutionFlags();
	List<const char*> *getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel);
	virtual List<DependencyArc*> *getAllTaskDependencies();
	List<const char*> *getVariablesNeedingCommunication(int segmentedPPS);
	List<CommunicationCharacteristics*> *getCommCharacteristicsForSyncReqs(int segmentedPPS);

	// this function is used by only the first composite stage (that represents the entire computation) and
	// repeat cycles to initiate the counters that are used to track if an updater of need-to-be synchronized
	// variable indeed did executed 
	void declareSynchronizationCounters(std::ofstream &stream, int indentation, int nestingLevel);
	
	// Just like data dependencies that are dragged down to PPU level, shared data update signals need to be
	// brought to PPU level to keep the scheme consistent. The following three methods are used for that purpose
	static List<SyncRequirement*> *getUpdateSignalsOfGroup(List<FlowStage*> *group);
	void generateDataSendsForGroup(std::ofstream &stream, int indentation,
			List<SyncRequirement*> *commRequirements);
	void generateSignalCodeForGroupTransitions(std::ofstream &stream, int indentation,
			List<SyncRequirement*> *syncRequirements);

	// Part of the synchronization scheme is to ensure that a modifier of data does not go too far ahead of the
	// reader stages so that the former has modified data several times before the latter get to finish reading.
	// Therefore, there is a need to model and resolve write-after-read dependencies. Information about the last
	// reader of a data -- the stage that should -- reactivate the writer for another possible update is there
	// in the dependency arcs. Arcs that are marked as reactivator should be used by the waiting stage to notify
	// the modifier stage that it can proceed; vice versa the modifier stage should wait for clearance of these
	// arc's signals. Consequently we have the following two methods for group reactivations.
	void generateCodeForWaitingForReactivation(std::ofstream &stream, int indentation, 
			List<SyncRequirement*> *syncRequirements);
	void generateCodeForReactivatingDataModifiers(std::ofstream &stream, int indentation, 
			List<SyncRequirement*> *syncDependencies);	
	void setReactivatorFlagsForSyncReqs();

	// These two are synchronization simplification functions for our initial barrier based implementation of
	// sync primitives. When we use barrier then our detail method for 
	// signalUpdate-waitForUpdate-signalRead-waitForRead sync cycles that offers the utmost flexibility for
	// computation and communication overlap boils down to lock-step bulk synchronous mode of execution. In that
	// case we do not need all four signals rather updaters should just waitForRead signals from readers to know
	// that last change is no longer needed then execute its code and signalUpdate.
	void genSimplifiedWaitingForReactivationCode(std::ofstream &stream, int indentation, 
			List<SyncRequirement*> *syncRequirements);
	void genSimplifiedSignalsForGroupTransitionsCode(std::ofstream &stream, int indentation,
			List<SyncRequirement*> *syncRequirements);
	
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
	void performEpochUsageAnalysis();
	void calculateLPSUsageStatistics();
	List<DependencyArc*> *getAllTaskDependencies();
	void generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace);
	int assignIndexAndGroupNo(int currentIndex, int currentGroupNo, int currentRepeatCycle);
	void setLpsExecutionFlags();
	
	// This function indicates if the repeat loop condition evaluation includes any LPS dependent varialbe. If
	// it does not then the repeat loop can be lifted up and can be executed above the LPS indicated by its 
	// space variable. This result in the distinction between having an LPU iteration (due to multiplexing of
	// LPUs into PPUs) inside the repeat loop as opposed to outside it.
	bool isLpsDependent();

	// this is a helper routine for code generation to temporarily change the LPS of a repeat cycle as needed
	void changeSpace(Space *newSpace) { this->space = newSpace; }
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
