#ifndef _H_lpu_management
#define _H_lpu_management

#include "structure.h"
#include "../utils/list.h"
#include <fstream>

/* Remember that there is a partial ordering of logical processing spaces (LPS). Thereby, the number of
   LPUs for a child LPS at a particular point of computation depends on the size of the data structure
   partitions defined by current LPUs of the ancester LPSes. Thus, the range of LPUs a particular thread
   will execute from a particular LPS varies at runtime.

   For the same reason, the mechanism for generating the next LPU for an LPS is a recursive procedure.
   Depending on the scenario, if a thread exhausts all LPUs of the current LPS it is doing some 
   computation on, it may just declare that there is nothing more to compute; or it may go recursively
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
	virtual int *copyCompositeLpuId();
	virtual int *setCurrentCompositeLpuId(int linearId);
	int getCurrentLpuId() { return currentLinearLpuId; }
	virtual int getNextLpuId(int previousLpuId);
	virtual void resetCounter();
	virtual void logLpuRange(std::ofstream &log, int indent);
	virtual void logLpuCount(std::ofstream &log, int indent);
	virtual void logCompositeLpuId(std::ofstream &log, int indent);
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
	int *copyCompositeLpuId();
	int *setCurrentCompositeLpuId(int linearId);
	int getNextLpuId(int previousLpuId);
	void resetCounter() { currentLinearLpuId = INVALID_ID; }
	void logLpuRange(std::ofstream &log, int indent) {}
	void logLpuCount(std::ofstream &log, int indent);
	void logCompositeLpuId(std::ofstream &log, int indent);
};

/* base class for LPUs of all LPSes; task specific subclasses will add other necessary fields  */
class LPU {
  public:
	int id;
	bool valid;
	
	LPU() { id = 0; valid = false; } 
	void setId(int id) { this->id = id; }	
	void setValidBit(bool valid) { this->valid = valid; }
	bool isValid() { return valid; }
	virtual void print(std::ofstream &stream, int indentLevel) {}
};

/* base class for task metadata object that holds the dimension information of all arrays been used */
class Metadata {
  public:
	const char *taskName;
	Metadata() { taskName = NULL; }
	void setTaskName(const char *taskName) { this->taskName = taskName; }
	const char *getTaskName() { return taskName; }
	virtual void print(std::ofstream &stream) {}
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
	LPU *getCurrentLpu();
	void invalidateCurrentLpu() { lpu->setValidBit(false); }
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
	virtual void setRootLpu(Metadata *metadata) = 0;
	virtual void setRootLpu(LPU *rootLpu) = 0;
	virtual void initializeLPUs() = 0;
	virtual int *computeLpuCounts(int lpsId) = 0;
	virtual LPU *computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) = 0;

	// The get-Next-Lpu management routine is at the heart of recursive LPU management for threads by
	// the runtime library. It takes as input the ID of the LPS on which the thread is attempting to
	// execute any compute-stage and the ID of the LPS from which the flow of control descended to current
	// LPS under concern. Alongside, it takes input the ID of the last LPU been executed in current LPS
	// to distinguish between first-time and subsequent requests. Internally, it runs a recursive 
	// procedure to set up LPUs on not only the current LPS but also any LPS in-between the container 
	// and the current. Furthermore, it maintains the state of those LPSes as computation continues on
	// LPUs after LPUs. It returns NULL when the recursive process has no more LPUs to return.	
	LPU *getNextLpu(int lpsId, int containerLpsId, int currentLpuId);

	// Following two routines are added to aid memory management in segmented memory system. The idea here 
	// is to get the Ids of all LPUs that are multiplexed to a thread before it begin executions. A 
	// segmented-PPU controller then accumulates all these Ids and passes them as a part of initialization 
	// arguments to the memory management module. Once memory has been allocated for all data structure 
	// parts needed for different LPUs and they have been scheduled, the segmented-PPU controller can 
	// launch its threads. Then each thread will proceed just like it does in the case of multicore CPU 
	// backends.
	int getNextLpuId(int lpsId, int containerLpsId, int currentLpuId);
	// Note that for each LPU, we need to return not only the -- possibly multidimensional -- LPU id but
	// also the ids of its ancestor LPUs in upper LPSes. This is because, LPU ids are hierarchical and
	// sizes of different data parts in a lower LPS varies depending on the size of their ancestor parts.
	List<List<int*>*> *getAllLpuIds(int lpsId, int rootLpsId);

	LPU *getCurrentLpu(int lpsId);
	void removeIterationBound(int lpsId);
	ThreadIds *getThreadIds() { return threadIds; }
	bool isValidPpu(int lpsId);
	void initiateLogFile(const char *fileNamePrefix);
	int getThreadNo() { return threadIds->threadNo; }
	virtual ~ThreadState() {}
	
	// a log file for diagnostics
	std::ofstream threadLog;
	void logExecution(const char *stageName, int spaceId);
	void logThreadAffinity();
	void closeLogFile();
};


#endif
