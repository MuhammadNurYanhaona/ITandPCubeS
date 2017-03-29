#ifndef _H_sync_stage_implant
#define _H_sync_stage_implant

#include "../common/constant.h"
#include "../semantics/scope.h"
#include "../semantics/task_space.h"
#include "../semantics/data_access.h"
#include "../semantics/computation_flow.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

/*-------------------------------------------------------------------------------------------------------------------------      
	Sync stages are automatically added to the user specified computation flow during static analysis. Sync 
        stages have no code within. They only keep track of the data structures that need to be synchronized. This 
	header file contains all the classes associated with sync-stage implantations within the computation flow. 
-------------------------------------------------------------------------------------------------------------------------*/

class TaskEnvStat;

/* 	the definition of a sync stage */
class SyncStage : public FlowStage {
  protected:
        SyncMode mode;
        SyncStageType type;
	
	// descriptive name for the sync stage for printing purpose
	const char *name;
  public:
        SyncStage(Space *space, SyncMode mode, SyncStageType type);
	void setName(const char *name) { this->name = name; }
	void print(int indent);
        int populateAccessMap(List<VariableAccess*> *accessLogs, 
		bool filterOutNonReads, bool filterOutNonWritten);
	void addAccessInfo(VariableAccess *accessLog);
        bool isLoaderSync() { return (mode == Load || mode == Load_And_Configure); }
	void performDataAccessChecking(Scope *taskScope) {}
	void populateAccessMapForSpaceLimit(Hashtable<VariableAccess*> *accessMapInProgress,
                        Space *lps, bool includeLimiterLps) {}
	void calculateLPSUsageStatistics() {}
	void fillInTaskEnvAccessList(List<VariableAccess*> *envAccessList) {}
        void prepareTaskEnvStat(TaskEnvStat *taskStat) {}
};

/*      This is a utility class to keep track of the last point of entry to a space as flow of control moves from
        flow stages to flow stages. This is required so that during a space exit we can determine what previous 
        stages to check for potential data structure changes that we may need to synchronize. A reference to any
        possible sync stage associated with space entry is also maintained to be updated at space exit. This is 
	done to read only those data structures that are been accessed in actual computation in between entry and 
	exit. 
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

/*      This is a utility class that checks space transitions as the flow of control moves from flow stages to
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

/*      This is merely an utility class to keep track what data structure should be synchronized to what ancestor 
	space when flow of control exits from a space whose data structures wont last after the exit.
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
