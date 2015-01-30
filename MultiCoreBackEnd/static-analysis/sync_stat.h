#ifndef _H_sync_stat
#define _H_sync_stat

#include <iostream>
#include <fstream>

#include "../semantics/task_space.h"
#include "data_flow.h"
#include "data_access.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

/* This header file comprises classes that are needed to extract synchronization requirements from 
   dependency arcs. For each flow-stage we should have all synchronization needs encoded once it
   has been executed. Note that the information produced here is still in abstract from. Later during
   back-end compiler phases when we get the mapping information and thereby know the PPU ids that
   will be responsible for executing the various execution stages of a task, we will know how to 
   choose actual synchronization primitives matching the requirements of the task. 
*/

// This is the common super-class to encode the sync requirement to a single computation stage in a 
// single LPS due to a change of a single variable. Note  that the change can happen in the same LPS
// the dependent computation resides in.  
class SyncRequirement {
  protected:
	const char *syncTypeName;
	const char *variableName;
	Space *dependentLps;
	FlowStage *waitingComputation;
	DependencyArc *arc;
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
	bool isActive() { return arc->isActive(); }
	void deactivate() { arc->deactivate(); }	
	virtual void print(int indent);
	void writeDescriptiveComment(std::ofstream &stream, bool forDependent);		
};

// As the name suggests, this represents a sync requirements among all LPUs within an LPS due to a
// replicated variable update within one of them
class ReplicationSync : public SyncRequirement {
  public:	
	ReplicationSync() : SyncRequirement("Replication") {}
	void print(int indent);		
};

// This class indicates an overlapping padding (a.k.a. ghost) region syncing among adjacent LPUs of a
// single LPS
class GhostRegionSync : public SyncRequirement {
  protected:
	List<int> *overlappingDirections;
  public:
	GhostRegionSync();
	void setOverlappingDirections(List<int> *overlappingDirections);	
	void print(int indent);		
};

// This class is to indicate that a change in some variable in a lower LPS by all LPUs of that LPS needs
// to be synchronized in an ancestor LPS.
class UpPropagationSync : public SyncRequirement {
  public:	
	UpPropagationSync() : SyncRequirement("Gather/Reduction") {}
	void print(int indent);		
};

// This does the exact opposite of the previous computation
class DownPropagationSync : public SyncRequirement {
  public:	
	DownPropagationSync() : SyncRequirement("Broadcast/Scatter") {}
	void print(int indent);		
};

// Cross propagation synchronizations are needed when a variable is shared by two LPSes that are not 
// hierarchically related to each other and is been modified in one of them. This requires determining
// what LPU in one LPS depends on what on the other LPS. Presence of dynamic LPSes complicate this 
// calculation and we may have to extend this class further in the future.
class CrossPropagationSync : public SyncRequirement {
  public:
	CrossPropagationSync() : SyncRequirement("Redistribution") {}
	void print(int indent);		
};

// This class holds all synchronization requirements due to a single variable update within a computation
// stage.
class VariableSyncReqs {
  protected:
	const char *varName;
	List<SyncRequirement*> *syncList;
  public:
	VariableSyncReqs(const char *varName);
	void addSyncRequirement(SyncRequirement *syncReq);
	List<SyncRequirement*> *getSyncList() { return syncList; }    
	void print(int indent);		
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
	void addVariableSyncReq(const char *varName, SyncRequirement *syncReq);
	VariableSyncReqs *getVarSyncReqs(const char *varName) { return varSyncMap->Lookup(varName); }
	List<VariableSyncReqs*> *getVarSyncList();
	bool isDependentStage(FlowStage *suspectedDependentStage);
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
