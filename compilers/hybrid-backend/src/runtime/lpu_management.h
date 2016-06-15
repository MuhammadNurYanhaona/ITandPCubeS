#ifndef _H_lpu_management
#define _H_lpu_management

#include "structure.h"
#include "../memory-management/part_tracking.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_management.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include <fstream>

class Communicator;

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
	LPU *getCurrentLpu(bool allowInvalid = false);
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
	// a map to keep track of all partition configurations used for data structures in various LPSes
	Hashtable<DataPartitionConfig*> *partConfigMap;
	// a reference to the instance that holds all memory allocated for the task
	TaskData *taskData;
	// a map of iterators that will be needed to identifiy data parts for LPUs
	Hashtable<PartIterator*> *partIteratorMap;
	// a map of communicators to be used to exchange data for data dependencies involving communication
	Hashtable<Communicator*> *communicatorMap;
	// an auxiliary variable used in LPU generation process
	List<int*> *lpuIdChain;
  public:
	ThreadState(int lpsCount, int *lpsDimensions, int *partitionArgs, ThreadIds *threadIds);
	
	ThreadIds *getThreadIds() { return threadIds; }
	void setPartConfigMap(Hashtable<DataPartitionConfig*> *map) { partConfigMap = map; }
	Hashtable<DataPartitionConfig*> *getPartConfigMap() { return partConfigMap; }
	void setTaskData(TaskData *taskData) { this->taskData = taskData; }
	TaskData *getTaskData() { return taskData; }
	void setPartIteratorMap(Hashtable<PartIterator*> *map) { this->partIteratorMap = map; }
	PartIterator *getIterator(int lpsId, const char *varName);
	void setCommunicatorMap(Hashtable<Communicator*> *map) { this->communicatorMap = map; }
	Communicator *getCommunicator(const char *dependencyName) { 
		return communicatorMap->Lookup(dependencyName); 
	}
			
	virtual void setLpsParentIndexMap() = 0;
	virtual void setRootLpu(Metadata *metadata) = 0;
	virtual void setRootLpu(LPU *rootLpu) = 0;
	virtual void initializeLPUs() = 0;
	virtual int *computeLpuCounts(int lpsId) = 0;
	virtual LPU *computeNextLpu(int lpsId) = 0;

	// The get-Next-Lpu management routine is at the heart of recursive LPU management for threads by
	// the runtime library. It takes as input the ID of the LPS on which the thread is attempting to
	// execute any compute-stage and the ID of the LPS from which the flow of control descended to current
	// LPS under concern. Alongside, it takes input the ID of the last LPU been executed in current LPS
	// to distinguish between first-time and subsequent requests. Internally, it runs a recursive 
	// procedure to set up LPUs on not only the current LPS but also any LPS in-between the container 
	// and the current. Furthermore, it maintains the state of those LPSes as computation continues on
	// LPUs after LPUs. It returns NULL when the recursive process has no more LPUs to return.	
	LPU *getNextLpu(int lpsId, int containerLpsId, int currentLpuId);

	// The following routine is added to aid memory management in segmented memory system. The idea here 
	// is to get the Ids of all LPUs that are multiplexed to a thread before it begin executions. A 
	// segmented-PPU controller then accumulates all these Ids and passes them as a part of initialization 
	// arguments to the memory management module. Once memory has been allocated for all data structure 
	// parts needed for different LPUs and they have been scheduled, the segmented-PPU controller can 
	// launch its threads. Then each thread will proceed just like it does in the case of multicore CPU 
	// backends.
	int getNextLpuId(int lpsId, int containerLpsId, int currentLpuId);

	// function to be used at runtime to propel LPU creation from ID; this is a makeshift operation to
	// reduce the amount of changes we need to make in our transition from multicore to segmented-memory
	// backends; there should be some better way to generate the LPUs hierarchically from configurations
	//
	// Note that these functions assume that there is currently a valid LPU for the LPS represented by the
	// first argument.
	int *getCurrentLpuId(int lpsId);
	// Note that for each LPU, we need to return not only the -- possibly multidimensional -- LPU id but
	// also the ids of its ancestor LPUs in upper LPSes. This is because, LPU ids are hierarchical and
	// sizes of different data parts in a lower LPS varies depending on the size of their ancestor parts.
	List<int*> *getLpuIdChain(int lpsId, int rootLpsId);
	// it is just like the previous function but does not create in objects in the memory in the process;
	// rather it just return a list formed by ids taken from different LPS counters; it should be used 
	// with caution
	List<int*> *getLpuIdChainWithoutCopy(int lpsId, int rootLpsId);
	

	// returns the current multidimensional LPU count stored in the state-counter of the LPS indicated by 
	// the argument; note that this function assumes that the count has been computed already 
	int *getLpuCounts(int lpsId);

	LPU *getCurrentLpu(int lpsId, bool allowInvalid = false);
	void removeIterationBound(int lpsId);
	bool isValidPpu(int lpsId);
	int getThreadNo() { return threadIds->threadNo; }
	virtual ~ThreadState() {}
	
	// a log file for diagnostics and corresponding methods
	std::ofstream threadLog;
	bool loggingEnabled;
	void logExecution(const char *stageName, int spaceId);
	void logThreadAffinity();
	void closeLogFile();
	void enableLogging() { loggingEnabled = true; }
	void initiateLogFile(const char *fileNamePrefix);
	void logIteratorStatistics();
};

/* This is the class to hold the PPU execution controllers (here threads) that shares a single memory segment */
class SegmentState {
  protected:
	// logical Id of the segment
	int segmentId;
	// the id to be used to communicate between segments
	int physicalId;
	// state of the threads that are parts of a segment
	List<ThreadState*> *participantList;
	// the partition configuration object to be used to generate interval descriptions for data from LPU ids
	Hashtable<DataPartitionConfig*> *partConfigMap;
  public:
	SegmentState(int segmentId, int physicalId);
	int getSegmentId() { return segmentId; }
	int getPhysicalId() { return physicalId; }
	void setPartConfigMap(Hashtable<DataPartitionConfig*> *partConfigMap) { 
		this->partConfigMap = partConfigMap; 
	}
	void addParticipant(ThreadState *thread) { participantList->Append(thread); }
	List<ThreadState*> *getParticipantList() { return participantList; }

	// this tells how many participants will do computations suppossed to execute within a particular LPS
	int getPpuCountForLps(int lpsId);
	
	bool computeStagesInLps(int lpsId) { return getPpuCountForLps(lpsId) > 0; }
};

#endif
