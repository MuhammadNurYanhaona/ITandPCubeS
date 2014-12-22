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
	FlowStage *source;
	FlowStage *destination;
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
  public:
	DependencyArc(FlowStage *source, FlowStage *destination, const char *varName);
	FlowStage *getSource() { return source; }
	FlowStage *getDestination() { return destination; }
	const char *getVarName() { return varName; }
	bool isActive() { return active; }
	void activate() { active = true; }
	void deactivate() { active = false; }
	void print(int indent, bool displaySource, bool displayDestination);
	void deriveSyncAndCommunicationRoots(PartitionHierarchy *hierarchy);
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
};

#endif
