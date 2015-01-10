/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "coo_matrix_vector_multiplication.h"

/*-----------------------------------------------------------------------------------
header files included for different purposes
------------------------------------------------------------------------------------*/
// for error reporting and diagnostics
#include <iostream>
#include <string>
#include <cstdlib>

// for math functions
#include <math.h>

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

using namespace coomvm;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *coomvm::getLPUsCountOfSpaceA(int ppuCount, Dimension w_localDim1) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(w_localDim1, ppuCount, 1);
	return count;
}

int *coomvm::getLPUsCountOfSpaceB(int ppuCount, Dimension wDim1, int r) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(wDim1, ppuCount, r);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **coomvm::getmPartForSpaceALpu(PartitionDimension **mParentLpuDims, 
		int *lpuCount, int *lpuId, int p) {
	PartitionDimension **mLpuDims = new PartitionDimension*[1];
	mLpuDims[0] = new PartitionDimension;
	mLpuDims[0]->storageDim = mParentLpuDims[0]->partitionDim;
	mLpuDims[0]->partitionDim = block_count_getRange(*mParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], p, 0, 0);
	return mLpuDims;
}

PartitionDimension **coomvm::getw_localPartForSpaceALpu(PartitionDimension **w_localParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **w_localLpuDims = new PartitionDimension*[2];
	w_localLpuDims[0] = new PartitionDimension;
	w_localLpuDims[0]->storageDim = w_localParentLpuDims[0]->partitionDim;
	w_localLpuDims[0]->partitionDim = block_size_getRange(*w_localParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	w_localLpuDims[1] = w_localParentLpuDims[1];
	return w_localLpuDims;
}

PartitionDimension **coomvm::getwPartForSpaceBLpu(PartitionDimension **wParentLpuDims, 
		int *lpuCount, int *lpuId, int r) {
	PartitionDimension **wLpuDims = new PartitionDimension*[1];
	wLpuDims[0] = new PartitionDimension;
	wLpuDims[0]->storageDim = wParentLpuDims[0]->partitionDim;
	wLpuDims[0]->partitionDim = block_size_getRange(*wParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], r, 0, 0);
	return wLpuDims;
}

PartitionDimension **coomvm::getw_localPartForSpaceBLpu(PartitionDimension **w_localParentLpuDims, 
		int *lpuCount, int *lpuId, int r) {
	PartitionDimension **w_localLpuDims = new PartitionDimension*[2];
	w_localLpuDims[0] = w_localParentLpuDims[0];
	w_localLpuDims[1] = new PartitionDimension;
	w_localLpuDims[1]->storageDim = w_localParentLpuDims[1]->partitionDim;
	w_localLpuDims[1]->partitionDim = block_size_getRange(*w_localParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], r, 0, 0);
	return w_localLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *coomvm::getPpuIdsForThread(int threadNo)  {

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

	// for Space B;
	threadCount = Total_Threads;
	groupSize = threadCount / 1;
	groupThreadId = idsArray[Space_Root] % groupSize;
	threadIds->ppuIds[Space_B].groupId = idsArray[Space_Root] / groupSize;
	threadIds->ppuIds[Space_B].ppuCount = 1;
	threadIds->ppuIds[Space_B].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_B].id
			= threadIds->ppuIds[Space_B].groupId;
	else threadIds->ppuIds[Space_B].id = INVALID_ID;
	idsArray[Space_B] = groupThreadId;

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
	lpsParentIndexMap[Space_B] = Space_Root;
}

// Construction of task specific root LPU
void ThreadStateImpl::setRootLpu() {
	SpaceRoot_LPU *lpu = new SpaceRoot_LPU;
	lpu->m = NULL;
	lpu->mPartDims = new PartitionDimension*[1];
	lpu->mPartDims[0] = new PartitionDimension;
	lpu->mPartDims[0]->storageDim = lpu->mPartDims[0]->partitionDim
			= &arrayMetadata.mDims[0];

	lpu->v = NULL;
	lpu->vPartDims = new PartitionDimension*[1];
	lpu->vPartDims[0] = new PartitionDimension;
	lpu->vPartDims[0]->storageDim = lpu->vPartDims[0]->partitionDim
			= &arrayMetadata.vDims[0];

	lpu->w = NULL;
	lpu->wPartDims = new PartitionDimension*[1];
	lpu->wPartDims[0] = new PartitionDimension;
	lpu->wPartDims[0]->storageDim = lpu->wPartDims[0]->partitionDim
			= &arrayMetadata.wDims[0];

	lpu->w_local = NULL;
	lpu->w_localPartDims = new PartitionDimension*[2];
	lpu->w_localPartDims[0] = new PartitionDimension;
	lpu->w_localPartDims[0]->storageDim = lpu->w_localPartDims[0]->partitionDim
			= &arrayMetadata.w_localDims[0];
	lpu->w_localPartDims[1] = new PartitionDimension;
	lpu->w_localPartDims[1]->storageDim = lpu->w_localPartDims[1]->partitionDim
			= &arrayMetadata.w_localDims[1];

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
				*spaceRootLpu->w_localPartDims[0]->partitionDim);
	}
	if (lpsId == Space_B) {
		int ppuCount = threadIds->ppuIds[Space_B].ppuCount;
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceB(ppuCount, 
				*spaceRootLpu->wPartDims[0]->partitionDim, 
				partitionArgs[1]);
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
				spaceRootLpu->mPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->v = NULL;
		currentLpu->vPartDims = spaceRootLpu->vPartDims;
		currentLpu->w_local = NULL;
		currentLpu->w_localPartDims = getw_localPartForSpaceALpu(
				spaceRootLpu->w_localPartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	if (lpsId == Space_B) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceB_LPU *currentLpu = new SpaceB_LPU;
		currentLpu->w = NULL;
		currentLpu->wPartDims = getwPartForSpaceBLpu(
				spaceRootLpu->wPartDims, lpuCounts, nextLpuId, 
				partitionArgs[1]);
		currentLpu->w_local = NULL;
		currentLpu->w_localPartDims = getw_localPartForSpaceBLpu(
				spaceRootLpu->w_localPartDims, lpuCounts, nextLpuId, 
				partitionArgs[1]);
		return currentLpu;
	}
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void coomvm::initializeTask(TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		COOMVMPartition partition) {

	arrayMetadata.mDims[0] = environmentLinks.mDims[0];
	arrayMetadata.vDims[0] = environmentLinks.vDims[0];
	arrayMetadata.wDims[0] = arrayMetadata.mDims[0];
	arrayMetadata.w_localDims[0].range.min = 0;
	arrayMetadata.w_localDims[0].range.max = partition.p - 1;
	arrayMetadata.w_localDims[1] = arrayMetadata.wDims[0];
}

