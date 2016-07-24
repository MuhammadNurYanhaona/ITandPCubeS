#ifndef _H_usage_statistic
#define _H_usage_statistic

/* This class is defined to aid in memory allocation decision which needs to be based on the usage
   of variables then their actual presence in the LPS partition definition. The functionality of this
   class could be achieved by adding some features in the VariableAccess class of data_access.h.
   That class's design is, however, geared towards dependency analysis; therefore, we define a new
   class that gives a cleaner interface. 
*/

class LPSVarUsageStat {
  protected:
	bool reduced;
	int accessCount;
	// this variable is maintained to make the logic of tracking access count work for dynamic
	// LPSes. Everytime we enter a conditional dynamic LPS computation, that may result in a new
	// allocation for a variable despite it been used multiple time in the LPS already. This is
	// because each of each access may be on different LPUs and there is no point allocating space
	// and copying data then in some target platform then.
	int maxUninterruptedAccesses;
	// this is a variable to aid in code generation that indicates memory has been allocated for
	// this variable in the space under concern
	bool allocated;
  public:
	LPSVarUsageStat() {
		reduced = false;
		accessCount = 0;
		maxUninterruptedAccesses = 0;
		allocated = false;
	}
	void flagReduced() { reduced = true; }
	bool isReduced() { return reduced; }
	void addAccess() { accessCount++; }
	void resetAccessCount() {
		if (accessCount > maxUninterruptedAccesses) {
			maxUninterruptedAccesses = accessCount;
		}
		accessCount = 0;
	}
	bool isAccessed() { 
		return accessCount > 0 || maxUninterruptedAccesses > 0; 
	}
	bool isMultipleAccess() {
		return accessCount > 1 || maxUninterruptedAccesses > 1;
	}
	void flagAllocated() { allocated = true; }
	bool isAllocated() { return allocated; }		
};

#endif
