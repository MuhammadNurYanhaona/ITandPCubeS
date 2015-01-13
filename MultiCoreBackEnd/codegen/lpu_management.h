#ifndef _H_lpu_management
#define _H_lpu_management

#include "structure.h"

/* Remember that there is a partial ordering of logical processing spaces (LPS). Thereby, the number of
   LPUs for a child LPS at a particular point of computation depends on the size of the data structure
   partitions defined by current LPUs of the ancester LPSes. Thus, the range of LPUs a particular thread
   will execute from a particular LPS varies at runtime.

   For the same reason, the mechanism for generating the next LPU for an LPS is a recursive procedure.
   Depending on the scenario, if a thread exhausts all LPUs of the current LPS it is doing some 
   computation on it may just declare that there is nothing more to compute; or it may go recursively
   up to reset the ancester LPUs and renew its list for current LPS and continue from there.

   Consequently, determining how to get and update LPU counts and LPU references is an involved procedure
   and following classes are defined to aid this process. Note that since the actual representation of
   an LPU depends on the specific task under concern. The entire process could not be handled using static
   library routines alone. Rather, task specific get-count and get-LPU functions are needed.

   Here the classes are defined in such a way that a large part of the LPU management complexity can be
   handled by this classes. We will only extend appropriate classes and implement some virtual functions
   in task specific way to plug in specific get-count and get-LPU routines in the logic.     	   
*/

/* class for managing the LPU range for a specific LPS to be executed by  a particular thread */
class LpuCounter {
  protected:
	// need to know the dimensionality of the LPS as internal LPU Ids are multidimensional
	int lpsDimensions;
	// there would be one count for each dimension of the LPS
	int *lpuCounts;
	// a variable for reducing calculation during translation between composite and linear LPU Ids
	int *lpusUnderDimensions;
	// a range variable that holds the translated linear LPU Id range of the thread
	LpuIdRange *currentRange;
	// the multidimensional id of most recently returned LPU
	int *currentLpuId;
	// linear equivalent of the multidimensional id
	int currentLinearLpuId;
	// a constructor to be utilized by subclasses
	LpuCounter();
  public:
	LpuCounter(int lpsDimensions);
	virtual void setLpuCounts(int *lpuCounts);
	virtual int *getLpuCounts() { return lpuCounts; }
	virtual void setCurrentRange(PPU_Ids ppuIds);
	virtual int *getCompositeLpuId() { return currentLpuId; }
	virtual int *setCurrentCompositeLpuId(int linearId);
	int getCurrentLpuId() { return currentLinearLpuId; }
	virtual int getNextLpuId(int previousLpuId);
	virtual void resetCounter();
};

class MockLpuCounter : public LpuCounter {
  protected:
	bool active;
  public:
	MockLpuCounter(PPU_Ids ppuIds);
	void setLpuCounts(int *lpuCounts) {}
	int *getLpuCounts() { return NULL; }
	void setCurrentRange(PPU_Ids ppuIds) {}
	int *getCompositeLpuId() { return &currentLinearLpuId; }
	int *setCurrentCompositeLpuId(int linearId);
	int getNextLpuId(int previousLpuId);
	void resetCounter() { currentLinearLpuId = INVALID_ID; }
};

/* base class for LPUs of all LPSes; task specific subclasses will add other necessary fields  */
class LPU {
  public: 
	int id;
};

/* class for holding all necessary state information for an LPS of a thread */
class LpsState {
  protected:
	LpuCounter *counter;
	// if an LPS is marked as an iteration bound then the recursive routine for get next LPU invoked 
	// for any descendent LPS will not progress further up from this point. By default the root LPS
	// will remain marked as the iteration bound at the beginning.
	bool iterationBound;
  public:
	// a current LPU reference to aid in calculating LPUs for descendent LPSes
	LPU *lpu;

	LpsState(int lpsDimensions, PPU_Ids ppuIds);
	void markAsIterationBound() { iterationBound = true; }
	bool isIterationBound() { return iterationBound; }
	void removeIterationBound() { iterationBound = false; }
	void setCurrentLpu(LPU *currentLpu) { lpu = currentLpu; }
	LPU *getCurrentLpu() { return lpu; }
	LpuCounter *getCounter() { return counter; }
};

/* This class represents the complete state of a thread for a particular task.  Task specific functions
   for computing LPU counts and next LPU are plugged into the execution logic by the generated sub-class
   that implements designated virtual functions. 
*/
class ThreadState {
  protected:
	// need to know the number of LPSes to initiate the LPS states array
	int lpsCount;
	// there will be one state variable par LPS of the task	
	LpsState **lpsStates;
	// an indexing map so that parent LPS state can be quickly identified using just array lookups
	int *lpsParentIndexMap;
	// PPU Ids the the thread holding the state variable
	ThreadIds *threadIds;
	// a reference to the partition arguments (all are integers) passed during task invocation is 
	// maintaines as these arguments are needed for LPU calculation	
	int *partitionArgs;
  public:
	ThreadState(int lpsCount, int *lpsDimensions, int *partitionArgs, ThreadIds *threadIds);
		
	virtual void setLpsParentIndexMap() = 0;
	virtual void setRootLpu() = 0;
	virtual int *computeLpuCounts(int lpsId) = 0;
	virtual LPU *computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) = 0;

	LPU *getNextLpu(int lpsId, int containerLpsId, int currentLpuId);
	void removeIterationBound(int lpsId);
	ThreadIds *getThreadIds() { return threadIds; }
	bool isValidPpu(int lpsId); 
	virtual ~ThreadState() {}
};


#endif
