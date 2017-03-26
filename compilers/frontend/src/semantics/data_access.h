#ifndef _H_data_access
#define _H_data_access

#include "../syntax/ast.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

class Scope;
class VariableSymbol;
class FlowStage;
class Space;
class PartitionHierarchy;

/* 	Access Flags class, as its content suggests, keeps track how a particular variable has been used within
	a particular context.
*/
class AccessFlags {
  protected:	
	bool read;
	bool write;
	bool redirect;
	bool reduce;
  public:
	AccessFlags() { read = write = reduce = redirect = false; }
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

#endif
