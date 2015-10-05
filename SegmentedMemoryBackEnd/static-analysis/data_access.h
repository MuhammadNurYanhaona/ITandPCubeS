#ifndef _H_data_access
#define _H_data_access

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../syntax/ast.h"

class Scope;
class VariableSymbol;
class ComputeStage;
class FlowStage;
class Space;
class PartitionHierarchy;

/* 	Access Flags class, as its content suggests, keeps track how a particular variable is been used within
	a particular context.
*/
class AccessFlags {
  protected:	
	bool read;
	bool write;
	bool redirect;
	bool reduce;
  public:
	AccessFlags() { read = write = reduce = false; }
	void flagAsRead() { read = true; }
	void flagAsRedirected() { redirect = true; }
	void flagAsWritten() { write = true; }
	void flagAsReduced() { reduce = true; }
	bool isRead() { return read; }
	bool isWritten() { return (write || redirect); }
	bool isReduced() { return reduce; }
	bool isRedirected() { return redirect; }
	void mergeFlags(AccessFlags *other);
	void printFlags();		
};

/*	Variable Access class is used to track the usage of task global variables within initialize and compute 
	stages of individual tasks. Later on this information is used to determine ordering dependencies among 
	stages, and synchronization and communication needs for data structure in between stage transitions. Use
	of metadata and content of data structures are traced separately to optimize data movements.
*/
class VariableAccess {
  protected:
	const char *varName;
	bool contentAccess;
	AccessFlags *contentAccessFlags;
	bool metadataAccess;
	AccessFlags *metadataAccessFlags;
	bool localAccess;
  public:
	VariableAccess(const char *varName);
	void markContentAccess();
	bool isContentAccessed() { return contentAccess; }
	AccessFlags *getContentAccessFlags() { return contentAccessFlags; }
	void markMetadataAccess();
	bool isMetadataAccessed() { return metadataAccess; }
	AccessFlags *getMetadataAccessFlags() { return metadataAccessFlags; }
	void mergeAccessInfo(VariableAccess *other);
	const char *getName() { return varName; }
	void printAccessDetail(int indent);
	bool isRead() { 
		return contentAccess && (contentAccessFlags->isRead() 
				|| contentAccessFlags->isReduced()); 
	}
	bool isModified() { return contentAccess && contentAccessFlags->isWritten(); }
	void markLocalAccess() { localAccess = true; }
	bool isLocalAccess() { return localAccess; }
};

/*	This class is used to store information about local variables within a computation stage that refer to
	some task global variables. This is needed to ensure that we can track access to task global varibles
	even if they are made indirectly through some local variables.
*/
class TaskGlobalReferences {
  protected:
	Scope *globalScope;
	Hashtable<const char*> *referencesToGlobals;
  public:
	TaskGlobalReferences(Scope *taskGlobalScope);
	void setNewReference(const char *localVarName, const char *globalVarName);
	bool doesReferToGlobal(const char *localVarName);
	bool isGlobalVariable(const char *name);
	VariableSymbol *getGlobalRoot(const char *localVarName);
};

/*	This class is used during dependency analysis across computation stages. It stores information regarding
	the last modifier of each task global variables as we jump from stages to stages and continue drawing
	dependency arcs.
*/
class LastModifierPanel {
  protected:
	Hashtable<FlowStage*> *stageList;
  public:
	static LastModifierPanel *getPanel() { return panel; }
	FlowStage *getLastModifierOfVar(const char *varName);
	void setLastModifierOfVar(FlowStage *stage, const char *varName);
  private:
	static LastModifierPanel *panel;
	LastModifierPanel();
};

class DependencyArc {
  protected:
	// original source and destination of the dependency relationship
	FlowStage *source;
	FlowStage *destination;
	// signaling source and destination to satisfy the dependency relationship (these may differ from the previous two
	// as signaling and waiting can happen at upper level composte stages containing the original source and destinations)
	FlowStage *signalSrc;
	FlowStage *signalSink;
	// This is a flag to resolve write-after-read dependencies, which works the opposite direction of the dependency arc. 
	// Only the last sink stage that is affected by the execution of the signal source should enable the signal that will
	// allow the source to proceed to next update. Note that this flag and the above to stages are only sensible in the 
	// context where the dependency arc ensues a synchronization requirement
	bool reactivator;

	bool active;
	const char *varName;
	
	// The communication root is the first common ancestor Space of the source and destination spaces of a dependency
	// arc. This is needed to determine how to physically communicate data from source to destination. To be more 
	// precise. The sender needs to consider (or compose) partition functions from root to the receiver's space to be
	// able to distribute its changes of underlying data structures to the places that need that update. 
	Space *communicationRoot;
	// Sync root is the topmost Space from/above the communication root that have replication in the partition of the 
	// underlying data structure. If there is no replication this should be null. This information is needed to impose
	// synchronization barriers and exclusive update restriction to critical regions (i.e., shared data structures). 
	Space *syncRoot;
	// This is a variable that is use to determine where should the primitives/variables related to this sync dependency
	// should be put. To give an example, if a stage inside a repeat loop is the source to a sink dependency on a stage
	// outside then the synchronization should take place after all iterations of the repeat loop.
	int nestingIndex;
	// Indicates that update signal has been issued for this dependency already; therefore, there is no need to further
	// consider it from the source side 
	bool signaled;
	// An Id to indicate the index of the arc in the source's list; it is used to determine the variable name correspond
	// to the arc 
	int arcId;
	// A name constructed from LPS and data structure information to create runtime variables for this dependency
	const char *arcName;
  public:
	DependencyArc(FlowStage *source, FlowStage *destination, const char *varName);
	FlowStage *getSource() { return source; }
	FlowStage *getDestination() { return destination; }
	void setSignalSrc(FlowStage *signalSrc) { this->signalSrc = signalSrc; }
	FlowStage *getSignalSrc() { return signalSrc; }
	void setSignalSink(FlowStage *signalSink) { this->signalSink = signalSink; }
	FlowStage *getSignalSink() { return signalSink; }
	void setReactivator(bool reactivator) { this->reactivator = reactivator; }
	bool isReactivator() { return reactivator; }
	const char *getVarName() { return varName; }
	bool isActive() { return active; }
	void activate() { active = true; }
	void deactivate() { active = false; }
	bool isSignaled() { return signaled; }
	void signal() { signaled = true; }
	void setNestingIndex(int nestingIndex) { this->nestingIndex = nestingIndex; }
	int getNestingIndex();
	void setArcId(int id) { this->arcId = id; }
	int getArcId() { return arcId; }
	const char *getArcName();
	void print(int indent, bool displaySource, bool displayDestination);
	void deriveSyncAndCommunicationRoots(PartitionHierarchy *hierarchy);
	Space *getCommRoot() { return communicationRoot; }
	Space *getSyncRoot() { return syncRoot; }
};

/*	DataDependencies Class is for storing all the incoming and outgoing dependency arcs of a single computation stage.
*/
class DataDependencies {
  protected:
	List<DependencyArc*> *incomingArcs;
	List<DependencyArc*> *outgoingArcs;
  public:
	DataDependencies();
	void addIncomingArcIfNotExists(DependencyArc *arc);
	void addOutgoingArcIfNotExists(DependencyArc *arc);
	List<DependencyArc*> *getActiveDependencies();
	List<DependencyArc*> *getOutgoingArcs() { return outgoingArcs; }
	void print(int indent);
	// If an update of a varible within a flow stage has after-effect that is exactly the same as that of another variable
	// then we can combine the two signals and issue a single signal instead. This function is used to do the redundancy
	// analysis and deactive any redundant signals. 
	void deactivateRedundantOutgoingArcs();
};

#endif
