#ifndef _H_task_env_stat
#define _H_task_env_stat

#include "data_access.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

class FlowStage;

/* To manage the environmental data structures of an IT program, we need to deduce how different tasks use the 
 * environmental data structures they have access to. This header file holds detail information about a task's 
 * use of the environmental data structures it has access to.   
 */

/* This class tells if the end of a task execution the view of a data structure from a particular LPs  becomes 
 * stale or remains fresh. */ 
class EnvVarLpsStat {
  protected:
	Space *lps;
	bool fresh;
  public:
	EnvVarLpsStat(Space *lps) {
		this->lps = lps;
		this->fresh = true;
	}
	void setStaleFreshMarker(bool state) { fresh = state; }
	bool isFresh() { return fresh; }
	Space *getLps() { return lps; }
};

/* This class tells whether an environmental data structure was read-only inside a task or has it been updated.
 * It also keeps track of the states of different LPSes that accesses/updates the data structure as part of the 
 * task execution. */
class EnvVarStat {
  protected:
	const char *varName;
	bool read;
	bool updated;
	Hashtable<EnvVarLpsStat*> *lpsStats;
  public:
	EnvVarStat(VariableAccess *accessLog);
	const char *getVarName() { return varName; }
	void initiateLpsUsageStat(Space *lps);
	void flagReadOnLps(Space *lps);
	void flagWriteOnLps(Space *lps);
	bool isRead() { return read; }
	bool isUpdated() { return updated; }
	bool hasStaleLpses();
	List<Space*> *getFreshLpses() { return getLpsesForState(true); }
	List<Space*> *getStaleLpses() { return getLpsesForState(false); }
  private:
	List<Space*> *getLpsesForState(bool fresh);	
};

/* This class holds state information of all environmental data structures of a single task to facilitate access.
*/
class TaskEnvStat {
  protected:
	Hashtable<EnvVarStat*> *varStatMap;
  public:
	TaskEnvStat(List<VariableAccess*> *accessMap, Space *rootLps);
	EnvVarStat *getVariableStat(const char *varName) { return varStatMap->Lookup(varName); }
	
	// We adopt the policy of keeping all LPSes of a task up-to-date with the latest modification of all data
	// structures they individually access at the end of the task's execution. To achieve that, we insert 
	// sync stages for LPSes at the end of the computation flow that read all stale data structures an LPS is
	// interested in. We do this stage insertation before the data dependency analysis kicks in and draws the
	// dependency arcs between stages.  
	List<FlowStage*> *generateSyncStagesForStaleLpses();
};

#endif
