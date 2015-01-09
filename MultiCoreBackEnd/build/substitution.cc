/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "substitution.h"

/*-----------------------------------------------------------------------------------
header files included for different purposes
------------------------------------------------------------------------------------*/
// for error reporting and diagnostics
#include <iostream>
#include <string>
#include <cstdlib>

// for tuple definitions found in the source code
#include "tuple.h"
#include <vector>

// for LPU and PPU management data structures
#include "../codegen/structure.h"
#include "../codegen/lpu_management.h"

// for utility routines
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"

// for routines related to partition functions
#include "../partition-lib/index_xform.h"
#include "../partition-lib/partition_mgmt.h"

using namespace s;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *s::getLPUsCountOfSpaceA(int ppuCount, Dimension mDim2) {
	int *count = new int[1];
	count[0] = stride_partitionCount(mDim2, ppuCount);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **s::getmPartForSpaceALpu(PartitionDimension **mParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **mLpuDims = new PartitionDimension*[2];
	mLpuDims[0] = mParentLpuDims[0];
	mLpuDims[1] = new PartitionDimension;
	mLpuDims[1]->storageDim = mParentLpuDims[1]->partitionDim;
	mLpuDims[1]->partitionDim = stride_getRange(*mParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return mLpuDims;
}

PartitionDimension **s::getvPartForSpaceALpu(PartitionDimension **vParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **vLpuDims = new PartitionDimension*[1];
	vLpuDims[0] = new PartitionDimension;
	vLpuDims[0]->storageDim = vParentLpuDims[0]->partitionDim;
	vLpuDims[0]->partitionDim = stride_getRange(*vParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return vLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *s::getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	int threadCount;
	int groupSize;
	int groupThreadId;

	// for Space A;
	threadCount = Total_Threads;
	groupSize = threadCount / 16;
	groupThreadId = idsArray[Space_Root] % groupSize;
	threadIds->ppuIds[Space_A].groupId = idsArray[Space_Root] / groupSize;
	threadIds->ppuIds[Space_A].ppuCount = 16;
	threadIds->ppuIds[Space_A].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_A].id
			= threadIds->ppuIds[Space_A].groupId;
	else threadIds->ppuIds[Space_A].id = INVALID_ID;
	idsArray[Space_A] = groupThreadId;

	return threadIds;
}

/*-----------------------------------------------------------------------------------
Thread-State implementation class for the task
------------------------------------------------------------------------------------*/

// Construction of task specific LPS hierarchy index map
void ThreadStateImpl::setLpsParentIndexMap() {
	lpsParentIndexMap = new int[Space_Count];
	lpsParentIndexMap[Space_Root] = INVALID_ID;
	lpsParentIndexMap[Space_A] = Space_Root;
}

// Construction of task specific root LPU
void ThreadStateImpl::setRootLpu() {
	SpaceRoot_LPU *lpu = new SpaceRoot_LPU;
	lpu->m = NULL;
	lpu->mPartDims = new PartitionDimension*[2];
	lpu->mPartDims[0] = new PartitionDimension;
	lpu->mPartDims[0]->storageDim = lpu->mPartDims[0]->partitionDim
			= &arrayMetadata.mDims[0];
	lpu->mPartDims[1] = new PartitionDimension;
	lpu->mPartDims[1]->storageDim = lpu->mPartDims[1]->partitionDim
			= &arrayMetadata.mDims[1];

	lpu->v = NULL;
	lpu->vPartDims = new PartitionDimension*[1];
	lpu->vPartDims[0] = new PartitionDimension;
	lpu->vPartDims[0]->storageDim = lpu->vPartDims[0]->partitionDim
			= &arrayMetadata.vDims[0];

	lpsStates[Space_Root]->lpu = lpu;
}

// Implementation of task specific compute-LPU-Count function 
int *ThreadStateImpl::computeLpuCounts(int lpsId) {
	if (lpsId == Space_Root) {
		return NULL;
	}
	if (lpsId == Space_A) {
		int ppuCount = threadIds->ppuIds[Space_A].ppuCount;
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceA(ppuCount, 
				*spaceRootLpu->mPartDims[1]->partitionDim);
	}
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_A) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
		currentLpu->m = NULL;
		currentLpu->mPartDims = getmPartForSpaceALpu(
				spaceRootLpu->mPartDims, lpuCounts, nextLpuId);
		currentLpu->v = NULL;
		currentLpu->vPartDims = getvPartForSpaceALpu(
				spaceRootLpu->vPartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void s::initializeTask(TaskGlobals taskGlobals, ThreadLocals threadLocals, 
		bool lower_triangular_system) {
	arrayMetadata.mDims[0] = environmentLinks.mDims[0];
	arrayMetadata.mDims[1] = environmentLinks.mDims[1];
	arrayMetadata.vDims[0] = environmentLinks.vDims[0];
}

