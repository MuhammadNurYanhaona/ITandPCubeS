#ifndef _H_reduction_result_track
#define _H_reduction_result_track

/* The classes in this header file are for facilitating storage, access, and update of result variables 
 * of non-task-global reductions. A task-global reduction produces a single result for the entire task.
 * Non-task-global reductions have different results for different LPUs. Hence, there is a need for LPU
 * ID based result management.	 
 */

#include "reduction.h"
#include "../../../../common-libs/utils/list.h"

#include <vector>

using namespace std;

/* superclass for forming the reduction results container hierarchy */
class ReductionResultContainer {
  protected:
	int lpsIndex;
	int lpuDimIndex;
	vector<int> idArray;
  public:
	ReductionResultContainer(int lpsIndex, int lpuDimIndex);
	virtual ~ReductionResultContainer() {}
	int getLpsIndexOfNextContainer(vector<int> *lpuIdDimensions);
	int getlpuDimIndexOfNextContainer(vector<int> *lpuIdDimensions);
	
	virtual void initiateResultVarforLpu(List<int*> *lpuId, 
			int remainingPositions, 
			vector<int> *lpuIdDimensions) = 0;

	virtual reduction::Result *retrieveResultForLpu(List<int*> *lpuId) = 0;
};

/* class representing an intermediate node in the reduction result container hierarchy */
class InterimReductionResultContainer : public ReductionResultContainer {
  protected:
	vector<ReductionResultContainer*> nextLevelContainers;
  public:
	InterimReductionResultContainer(int lpsIndex, int lpuDimIndex);
	~InterimReductionResultContainer();	
	void initiateResultVarforLpu(List<int*> *lpuId, 
			int remainingPositions, 
			vector<int> *lpuIdDimensions);
	reduction::Result *retrieveResultForLpu(List<int*> *lpuId);
};

/* class representing a leaf level node that holds a list of reduction result variables for LPUs */
class TerminalReductionResultContainer : public ReductionResultContainer {
  protected:
	vector<reduction::Result*> resultVariables;
  public:
	TerminalReductionResultContainer(int lpsIndex, int lpuDimIndex);
	~TerminalReductionResultContainer();		
	void initiateResultVarforLpu(List<int*> *lpuId, 
			int remainingPositions, 
			vector<int> *lpuIdDimensions);
	reduction::Result *retrieveResultForLpu(List<int*> *lpuId);
}; 

/* class that facilitates insertion and access of reduction result variables in the container hierarchy */
class ReductionResultAccessContainer {
  protected:
	vector<int> lpuIdDimensions;
	int idComponents;
	ReductionResultContainer *topLevelContainer;
  public:
	ReductionResultAccessContainer(vector<int> idDimensions);
	~ReductionResultAccessContainer();

	// accessor function to be used at the beginning of a task execution to create reduction result 
	// variables for individual LPUs
	void initiateResultForLpu(List<int*> *lpuId);

	// accessor function to be used during LPU preparation to retrieve LPU's reduction result variable
	reduction::Result *getResultForLpu(List<int*> *lpuId);
};

#endif
