#ifndef _H_task_env_stat
#define _H_task_env_stat

#include "data_access.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

/* To manage the environmental data structures of an IT program, we need to deduce how different tasks allocate
 * and use the environmental data structures they have access to. This header file holds detail information about 
 * a task's use of the environmental data structures it has access to.   
 */

/* This class tells if the end of a task execution a particular parts list (always associated with some LPS 
 * allocation) becomes stale or fresh. */ 
class EnvVarAllocationStat {
  protected:
	Space *lps;
	bool fresh;
  public:
	EnvVarAllocationStat(Space *lps) {
		this->lps = lps;
		this->fresh = true;
	}
	void setStaleFreshMarker(bool state) { fresh = state; }
	bool isFresh() { return fresh; }
	Space *getLps() { return lps; }
};

/* This class tells whether an environmental data structure was read-only inside a task or has it been updated.
 * It also keeps track of the states of different LPS allocations the task accesses/updates */
class EnvVarStat {
  protected:
	const char *varName;
	bool read;
	bool updated;
	Hashtable<EnvVarAllocationStat*> *lpsStats;
  public:
	EnvVarStat(VariableAccess *accessLog);
	void initiateLpsAllocationStat(Space *lps);
	void flagReadOnLps(Space *lps);
	void flagWriteOnLps(Space *lps);
	bool isRead() { return read; }
	bool isUpdated() { return updated; }
	bool hasStaleAllocations();
	List<Space*> *getFreshAllocatorLpses() { return getAllocatorLpsesForState(true); }
	List<Space*> *getStaleAllocatorLpses() { return getAllocatorLpsesForState(false); }
  private:
	List<Space*> *getAllocatorLpsesForState(bool fresh);	
};

/* This class holds state information of all environmental data structures of a single task to facilitate access.*/
class TaskEnvStat {
  protected:
	Hashtable<EnvVarStat*> *varStatMap;
  public:
	TaskEnvStat(List<VariableAccess*> *accessMap, Space *rootLps);
	EnvVarStat *getVariableStat(const char *varName) { return varStatMap->Lookup(varName); }
};

#endif
