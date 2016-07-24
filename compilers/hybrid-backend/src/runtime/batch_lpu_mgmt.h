#ifndef _H_batch_lpu_mgmt
#define _H_batch_lpu_mgmt

/* Generating LPUs for execution inside the GPU gives us some unique challenge because it is prohibitively expensive,
 * if not infeasible, to execute the recursive get-next-LPU routine inside GPU. So the strategy we adopted is that the
 * CPU host will generate the LPUs for PPUs inside the GPU card and offload them into the latter. Initially, we thought
 * that we can just call the get-next-LPU routine as many times as we need in the host to generate sufficient number 
 * of GPU LPUs. But that strategy does not work, as GPU LPUs may have localization dependencies (e.g., all LPUs of a
 * subpartition LPS needs to go to the same PPU), creating the problem that having enough LPUs to offload does not
 * necessarily mean having work for all PPUs that concurrently execute inside the GPU.
 *
 * This library provides a solution to the aforementioned problem. It implements the strategy that the CPU host runs
 * a batch of independent get-next-LPU routines whose internal data structures are initialized for different GPU PPUs
 * and returns a vector of LPUs at a time that are intended for those  GPU PPUs.	  
 */

#include "lpu_management.h"
#include "structure.h"
#include "../memory-management/part_tracking.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_management.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../gpu-offloader/gpu_code_executor.h"

#include <fstream>
#include <vector>

// these four constants are used to determine what PPU controller should participate in what way during generation of
// LPUs for the particular LPS the computation flow is currently in
#define LPU_GEN_STATE_CONTRIB_ACTIVE 1
#define LPU_GEN_STATE_NON_CONTRIB_ACTIVE 2
#define LPU_GEN_STATE_CONTRIB_DEPLATED 3
#define LPU_GEN_STATE_NON_CONTRIB_DEPLATED 4

class BatchPpuState {
  protected:
	// this represents the number of LPSes in the task
	int lpsCount;

	// Remember that PPSes of the hardware, just like the LPSes of the task, have a hierarchical relationship. This
	// hierarchy dictates what PPU should execute what LPU once LPS-to-PPS mapping has been done. Note that we just
	// need the count of PPUs active at different levels of LPS-to-PPS mapping -- not their IDs. This is because the
	// PPU hierarchy is symmetric and the IDs can be determined just from the counts. Also remember that PPSes are
	// ordered bottom up. So the lower PPS's group leader PPU count should appear earlier in the vector.
	std::vector<int> *groupLeaderPpuCounts;
	
	// a PPU controller's LPS stack is maintained in a thread-state variable; the following vector maintains stacks
	// of all PPU controllers that reside within a GPU
	std::vector<ThreadState*> *ppuStates;
	
	// The get-next-LPU routine that works when PPU controllers are operating independently is a recursive LPU
	// retrieval routine that manipulates the LPU stacks of those PPU controllers. The change in the LPU stack of
	// the PPU controller is hidden from the caller except for the one case where the caller is informed that there
	// is no more LPUs to operate on for the current LPS the caller has requested a new LPU for. The caller uses 
	// this information to move into the nest step of the computation flow of the task. If it keeps asking for more
	// LPUs even after the no-more-LPU flag is returned, that results in resetting the LPU stack at the beginning
	// and reiterating the LPUs again. Depending on the context the caller may decide to do either of the two.
	// In the batch execution model, however, we cannot let LPUs to be repeated from some PPUs but not for the others.
	// Rather, we want all PPU controllers should finish producing the LPUs for non-repeated case before any starts
	// repeating its LPUs, if needed of-course. To achieve that effect, we let the PPU controllers that have depleted
	// their LPUs to just wait and invoke the get-next-LPU routines on the others that still have more LPUs left.
	// The following property is used to keep track of the current states of the PPU controllers at different LPSes.       
	List<std::vector<int>*> *ppuLpuGenerationStatusForLPSes;

	// Just like in the case of an isolated PPU controller, where we maintain a single LPU instance and update its
	// properties to denote different LPUs, we maintain an LPU vector per LPS in the batch mode
	List<std::vector<LPU*>*> *lpuVectorsForLPSes;

	// A list of gpu-code-executors are used to offload LPU computations to the GPU at different context of a task's
	// computation flow. A map is used to store them to identify an executor by it's context ID.
	Hashtable<GpuCodeExecutor*> *gpuCodeExecutors;

	// these two variables are for debugging activities inside the class
	bool loggingEnabled;
	std::ofstream *logFile;
  public:
	// note that the thread-state list should have the states ordered from first PPU's state to the last PPU's state
	BatchPpuState(int lpsCount, 
			List<ThreadState*> *ppuStateList, 
			std::vector<int> *groupLeaderPpuCounts);
	~BatchPpuState();
	std::vector<ThreadState*> *getPpuStates() { return ppuStates; }
	void setGpuCodeExecutors(Hashtable<GpuCodeExecutor*> *executors) { gpuCodeExecutors = executors; }
	GpuCodeExecutor *getGpuExecutorForContext(const char *contextId) { 
		return gpuCodeExecutors->Lookup(contextId); 
	}
	
	// This is the function that manage get-next-LPU calls to the underlying PPU controllers and returns a vector 
	// of LPUs for the LPS indicated by the LPS ID. The container LPS ID parameter is used to restrict the change
	// of LPU stacks of the PPU controllers in the get-next-LPU process from the container to the current LPS. The
	// last argument is used to identify the last searched locations in the recursive LPU generation process for
	// different PPU controllers. This function returns an LPU vector of LPUs, some of them may be NULL, as long as 
	// there is at least one NON-NULL LPU in the vector. Otherwise, it returns a NULL vector. 
	std::vector<LPU*> *getNextLpus(int lpsId, int containerLpsId, std::vector<int> *currentLpuIds);

	// This function returns the current LPU counts for the LPS under concern of all PPU controllers. These counts
	// are needed to construct multidimensional LPU IDs from linear IDs in intra-GPU computation. The second 
	// parameter is used when the calling context ensures that the LPU counts of all participating PPU controllers
	// are the same. So the count value can be retrieved from just one of them. 
	std::vector<int*> *genLpuCountsVector(int lpsId, bool singleEntry);

	// As we need the IDs of most recently generated batch of LPUs to generate the next batch, the beginning of the
	// batch LPU generation for the LPS the caller has just entered into requires a specialized treatment. This
	// function is used by the caller during the entrance to initialize the LPU vector with invalid IDs so that the
	// getNextLpus routine of the above can be kick started properly
	void initLpuIdVectorsForLPSTraversal(int lpsId, std::vector<int> *lpuIdVector);

	// Sometimes we may want that the get-next-LPU recursion mechanism to work in a batch-synchronous way for the
	// participating PPUs. To elaborate, we may want to keep the ancestor LPS's LPUs fixed and iterate over all the
	// LPUs of the current LPS for all PPU controllers before any PPU controller can move up and change its ancestor
	// LPU. So this helper function is provided to enable/disable PPU controllers based on their LPU generation 
	// status in the ancestor LPS 
	void adjustLpuIdVector(int lpsId, std::vector<int> *lpuIdVector, 
                int ancestorLpsId, std::vector<LPU*> *ancestorLpuVector);

	// returns true if any of the PPU controller participating in the group is supposed to execute compute stages in
	// the LPS indicated by the argument
	bool hasValidPpus(int lpsId);

	// this function is used to reset the recursion restrictions (i.e., up-to what LPS the get-next-LPU recursion can 
	// update stack for search for new LPUs) on the LPU retrieval process 
	void removeIterationBound(int lpsId);

	// if the calling context wants just the list of LPUs without any information about what PPU controller generated
	// which, it can use the following static utility function to convert the LPU vector returned by getNextLpus() to
	// a list 
	static void covertLpuVectorToList(List<LPU*> *destinationList, std::vector<LPU*> *lpuVector);
 
	void enableLogging(std::ofstream *logFile);
	static void extractLpuIdsFromLpuVector(std::vector<int> *idVector, std::vector<LPU*> *lpuVector);
  private:
	// This function is used to initialize the indexes of the PPU controllers that will generate LPUs for a new LPS 
	// the caller has entered into.
	void resetActivePpuIndexes(int lpsId);
	
	// This function is used to reserve exact capacity to the LPU vectors for different LPSes
	void initializeLpuVectors(); 
};

#endif
