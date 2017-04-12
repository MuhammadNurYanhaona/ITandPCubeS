#ifndef _H_sync_stat
#define _H_sync_stat

/* This header file comprises classes that are needed to extract synchronization requirements from dependency 
   arcs. For each flow-stage we should have all synchronization needs encoded with it. Note that information 
   produced here is still in an abstract from. Later during back-end compiler phases when we get the mapping 
   information and thereby know the PPUs that are responsible for executing the various flow stages of a task, 
   we will know how to choose actual synchronization primitives matching the requirements of the task. 
*/

#include <iostream>
#include <fstream>

#include "data_dependency.h"
#include "../semantics/task_space.h"
#include "../semantics/computation_flow.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

class CommunicationCharacteristics;

// This is the common super-class to encode the sync requirement of a single computation stage in a single LPS 
// due to a change of a single variable. Note that the change can happen in the same LPS the dependent 
// computation resides in.  
class SyncRequirement {
  protected:
	const char *syncTypeName;
	const char *variableName;
	Space *dependentLps;
	FlowStage *waitingComputation;
	DependencyArc *arc;

	// Signaling for some sync requirements are redundant due to the existing of earlier signals from the
        // same source LPS to the same destination LPS for the same underlying data structures. Although the
        // signaling may be redundant but the receiving/waiting is not as the concerned sync requirements may
        // be active in different nesting levels. Thus we cannot get rid of sync requirements with redundant
        // signals. The code generation for such partially redundant syncs is tricky. We adopt the strategy 
        // of calling wait or receiving data on the earlier replacement sync whenever a sync with a redundant
        // signal is encountered. This property keeps track of the replacement sync to aid code generation. 
        SyncRequirement *replacementSync;

	// Most synchronizations/communications get issued due to the execution of some flow-stage that does
        // computations. A counter variable indicates that the flow-stage did execute and thus their is a need 
	// of data sync/communication with other stages in such a case. When the synchronization/communication
        // is due to some compiler injected sync-stage, however, their is no conditional execution and there
        // is no need for a counter variable either. Following variable distinguishes between these two cases.   
        bool counterRequirement;

	// This index variable is kept with the sync-requirement to facilitate a unique ID generation for any
	// counter associated with this requirement.
        int index;

  public:
	SyncRequirement(const char *syncTypeName);
	virtual ~SyncRequirement() {}
	void setVariableName(const char *varName) { this->variableName = varName; }
	const char *getVariableName() { return variableName; }
	void setDependentLps(Space *dependentLps) { this->dependentLps = dependentLps; }
	Space *getDependentLps()  { return dependentLps; }	
	void setWaitingComputation(FlowStage *computation) { this->waitingComputation = computation; }
	FlowStage *getWaitingComputation() { return waitingComputation; }
	void setDependencyArc(DependencyArc *arc) { this->arc = arc; }
	DependencyArc *getDependencyArc() { return arc; }
	void signal() { arc->signal(); }
	bool isActive() { return arc->isActive(); }
        void deactivate() { arc->deactivate(); }
	void setReplacementSync(SyncRequirement *other) { replacementSync = other; }
	SyncRequirement *getReplacementSync() { return replacementSync; }
	void setCounterRequirement(bool requirement) { this->counterRequirement = requirement; }
        bool getCounterRequirement() { return counterRequirement; }
	const char *getSyncName();
	virtual void print(int indent);
	void writeDescriptiveComment(std::ofstream &stream, bool forDependent);

	//------------------------------------------------------------- Common helper functions for Code Generation

	// This decides in which LPS should the sync primitive belong to and returns that LPS to the caller.
        virtual Space *getSyncOwner();

        // This decides the logical coverage of a sync operation starting from the sync-owner. This is not 
	// necessarily equals to the dependentLps property and vary with the specific type of synchronization. 
        virtual Space *getSyncSpan() { return dependentLps; }

        // This corresponds to a makeshift mechanism to protect from multiple updates taking place before all 
        // readers have finished reading the last update. Current implementation of synchronization
        // primitives do not have support for signaling back to updater that the underlying data can be 
	// modified again. Thus, we need another variable for each sync requirement to serve reader-to-updater 
	// signaling back.
        const char *getReverseSyncName();

	// This function generates information about the need and nature of communication for the sync
        // requirement. Its logic depends on the mapping of LPSes to PPSes and memory allocation decisions. 
        // So it should be called only after those steps have been completed. The argument represents the ID
	// of the PPS at and above which PPUs have disjoint memories.  
        virtual CommunicationCharacteristics *getCommunicationInfo(int segmentationPPS);

	void setIndex(int index) { this->index = index; }
        int getIndex() { return index; }

	// sorting of sync-requirement is needed during code generation to ensure uniform sequencing of send/
	// signal and receive/wait across PPU controllers
	static List<SyncRequirement*> *sortList(List<SyncRequirement*> *reqList);

	// a helper function for sorting sync requirements. It returns 0 if the other sync requirement is 
        // equivalent to current instace, -1 if the current instance less than the other, and finally 1 if it 
        // is greater then the other. 
        int compareTo(SyncRequirement *other);

        // this filters out two kinds of data dependencies from a single list into two separate lists for 
	// communication not-communication based implementation
        static void separateCommunicationFromSynchronizations(int segmentedPPS,
                        List<SyncRequirement*> *sourceList,
                        List<SyncRequirement*> *commList, List<SyncRequirement*> *syncList);
};

// As the name suggests, this represents a sync requirement among all LPUs within an LPS due to a replicated 
// variable update within one of them
class ReplicationSync : public SyncRequirement {
  public:	
	ReplicationSync() : SyncRequirement("RSync") {}
	void print(int indent);		
};

// This class indicates an overlapping padding (a.k.a. ghost) region syncing among adjacent LPUs of a single 
// LPS
class GhostRegionSync : public SyncRequirement {
  protected:
	List<int> *overlappingDirections;
  public:
	GhostRegionSync();
	void setOverlappingDirections(List<int> *overlappingDirections);	
	void print(int indent);		

	//------------------------------------------------------------- Common helper functions for Code Generation

        CommunicationCharacteristics *getCommunicationInfo(int segmentationPPS);
};

// This class is to indicate that a change in some variable in a lower LPS by all LPUs of that LPS needs to be 
// synchronized in an ancestor LPS.
class UpPropagationSync : public SyncRequirement {
  public:	
	UpPropagationSync() : SyncRequirement("USync") {}
	void print(int indent);		
	Space *getSyncSpan();
};

// This does the exact opposite of the previous computation
class DownPropagationSync : public SyncRequirement {
  public:	
	DownPropagationSync() : SyncRequirement("DSync") {}
	void print(int indent);		
};

// Cross propagation syncs are needed when a variable is shared by two LPSes that are not hierarchically related 
// to each other and the variable has been modified in one of them. This requires determining what LPU in one 
// LPS depends on what on the other LPS. Presence of dynamic LPSes complicate this calculation and we may have 
// to extend this class further in the future.
class CrossPropagationSync : public SyncRequirement {
  public:
	CrossPropagationSync() : SyncRequirement("CSync") {}
	void print(int indent);		
};

// This class holds all synchronization requirements due to a single variable update within a computation stage.
class VariableSyncReqs {
  protected:
	const char *varName;
	List<SyncRequirement*> *syncList;
  public:
	VariableSyncReqs(const char *varName);
	void addSyncRequirement(SyncRequirement *syncReq);
	List<SyncRequirement*> *getSyncList() { return syncList; }
	const char *getVarName() { return varName; }   
	void print(int indent);
	void deactivateRedundantSyncReqs();		
};

// This class holds synchronization requirements on different variables due to the execution of a single 
// computation stage.
class StageSyncReqs {
  protected:
	FlowStage *computation;
	Space *updaterLps;
	Hashtable<VariableSyncReqs*> *varSyncMap;
  public:
	StageSyncReqs(FlowStage *computation);
	
	// This function takes a boolean flag as the third argument. This is to indicate whether or not we want 
	// to add the dependency on the waiting flow-stage when updating the signaler flow stage which own this 
	// list.
	void addVariableSyncReq(const char *varName, SyncRequirement *syncReq, bool addDependency);

	VariableSyncReqs *getVarSyncReqs(const char *varName) { return varSyncMap->Lookup(varName); }
	List<VariableSyncReqs*> *getVarSyncList();
	bool isDependentStage(FlowStage *suspectedDependentStage);
	List<SyncRequirement*> *getAllSyncRequirements();
	List<SyncRequirement*> *getAllNonSignaledSyncReqs();
	void removeRedundencies();
	void print(int indent);		
};

// This class holds all synchronization dependencies into a flow stage (composite, computation, or data mover)
// due to execution of other stages
class StageSyncDependencies {
  protected:
	FlowStage *computation;
	Space *dependentLps;
	List<SyncRequirement*> *syncList;
  public:
	StageSyncDependencies(FlowStage *computation);
	void addDependency(SyncRequirement *syncReq);
	void addAllDependencies(List<SyncRequirement*> *syncReqList);
	List<SyncRequirement*> *getActiveDependencies();
	List<SyncRequirement*> *getDependencyList() { return syncList; }
};

#endif
