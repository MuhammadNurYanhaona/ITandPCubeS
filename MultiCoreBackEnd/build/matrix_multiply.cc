/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "matrix_multiply.h"

/*-----------------------------------------------------------------------------------
header files included for different purposes
------------------------------------------------------------------------------------*/
// for error reporting and diagnostics
#include <iostream>
#include <string>
#include <cstdlib>

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

using namespace mm;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *mm::getLPUsCountOfSpaceA(int ppuCount, Dimension cDim1, int k, Dimension cDim2, int l) {
	int *count = new int[2];
	count[0] = block_size_partitionCount(cDim1, ppuCount, k);
	count[1] = block_size_partitionCount(cDim2, ppuCount, l);
	return count;
}

int *mm::getLPUsCountOfSpaceA_Sub(int ppuCount, Dimension aDim2, int q) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(aDim2, ppuCount, q);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **mm::getaPartForSpaceALpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId, int k) {
	PartitionDimension **aLpuDims = new PartitionDimension*[2];
	aLpuDims[0] = new PartitionDimension;
	aLpuDims[0]->storageDim = aParentLpuDims[0]->partitionDim;
	aLpuDims[0]->partitionDim = block_size_getRange(*aParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], k, 0, 0);
	aLpuDims[1] = aParentLpuDims[1];
	return aLpuDims;
}

PartitionDimension **mm::getbPartForSpaceALpu(PartitionDimension **bParentLpuDims, 
		int *lpuCount, int *lpuId, int l) {
	PartitionDimension **bLpuDims = new PartitionDimension*[2];
	bLpuDims[0] = new PartitionDimension;
	bLpuDims[0]->storageDim = bParentLpuDims[0]->partitionDim;
	bLpuDims[0]->partitionDim = block_size_getRange(*bParentLpuDims[0]->partitionDim, 
			lpuCount[1], lpuId[1], l, 0, 0);
	bLpuDims[1] = bParentLpuDims[1];
	return bLpuDims;
}

PartitionDimension **mm::getcPartForSpaceALpu(PartitionDimension **cParentLpuDims, 
		int *lpuCount, int *lpuId, int k, int l) {
	PartitionDimension **cLpuDims = new PartitionDimension*[2];
	cLpuDims[0] = new PartitionDimension;
	cLpuDims[0]->storageDim = cParentLpuDims[0]->partitionDim;
	cLpuDims[0]->partitionDim = block_size_getRange(*cParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], k, 0, 0);
	cLpuDims[1] = new PartitionDimension;
	cLpuDims[1]->storageDim = cParentLpuDims[1]->partitionDim;
	cLpuDims[1]->partitionDim = block_size_getRange(*cParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], l, 0, 0);
	return cLpuDims;
}

PartitionDimension **mm::getaPartForSpaceA_SubLpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId, int q) {
	PartitionDimension **aLpuDims = new PartitionDimension*[2];
	aLpuDims[0] = aParentLpuDims[0];
	aLpuDims[1] = new PartitionDimension;
	aLpuDims[1]->storageDim = aParentLpuDims[1]->partitionDim;
	aLpuDims[1]->partitionDim = block_size_getRange(*aParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], q, 0, 0);
	return aLpuDims;
}

PartitionDimension **mm::getbPartForSpaceA_SubLpu(PartitionDimension **bParentLpuDims, 
		int *lpuCount, int *lpuId, int q) {
	PartitionDimension **bLpuDims = new PartitionDimension*[2];
	bLpuDims[0] = new PartitionDimension;
	bLpuDims[0]->storageDim = bParentLpuDims[0]->partitionDim;
	bLpuDims[0]->partitionDim = block_size_getRange(*bParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], q, 0, 0);
	bLpuDims[1] = bParentLpuDims[1];
	return bLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *mm::getPpuIdsForThread(int threadNo)  {

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

	// for Space A_Sub;
	threadIds->ppuIds[Space_A_Sub].groupId = 0;
	threadIds->ppuIds[Space_A_Sub].ppuCount = 1;
	threadIds->ppuIds[Space_A_Sub].groupSize = threadIds->ppuIds[Space_A].groupSize;
	threadIds->ppuIds[Space_A_Sub].id = 0;
	idsArray[Space_A_Sub] = idsArray[Space_A];

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
	lpsParentIndexMap[Space_A_Sub] = Space_A;
}

// Construction of task specific root LPU
void ThreadStateImpl::setRootLpu() {
	SpaceRoot_LPU *lpu = new SpaceRoot_LPU;
	lpu->a = NULL;
	lpu->aPartDims = new PartitionDimension*[2];
	lpu->aPartDims[0] = new PartitionDimension;
	lpu->aPartDims[0]->storageDim = lpu->aPartDims[0]->partitionDim
			= &arrayMetadata.aDims[0];
	lpu->aPartDims[1] = new PartitionDimension;
	lpu->aPartDims[1]->storageDim = lpu->aPartDims[1]->partitionDim
			= &arrayMetadata.aDims[1];

	lpu->b = NULL;
	lpu->bPartDims = new PartitionDimension*[2];
	lpu->bPartDims[0] = new PartitionDimension;
	lpu->bPartDims[0]->storageDim = lpu->bPartDims[0]->partitionDim
			= &arrayMetadata.bDims[0];
	lpu->bPartDims[1] = new PartitionDimension;
	lpu->bPartDims[1]->storageDim = lpu->bPartDims[1]->partitionDim
			= &arrayMetadata.bDims[1];

	lpu->c = NULL;
	lpu->cPartDims = new PartitionDimension*[2];
	lpu->cPartDims[0] = new PartitionDimension;
	lpu->cPartDims[0]->storageDim = lpu->cPartDims[0]->partitionDim
			= &arrayMetadata.cDims[0];
	lpu->cPartDims[1] = new PartitionDimension;
	lpu->cPartDims[1]->storageDim = lpu->cPartDims[1]->partitionDim
			= &arrayMetadata.cDims[1];

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
				*spaceRootLpu->cPartDims[0]->partitionDim, 
				partitionArgs[0], 
				*spaceRootLpu->cPartDims[1]->partitionDim, 
				partitionArgs[1]);
	}
	if (lpsId == Space_A_Sub) {
		int ppuCount = threadIds->ppuIds[Space_A_Sub].ppuCount;
		SpaceA_LPU *spaceALpu = (SpaceA_LPU*) 
				lpsStates[Space_A]->lpu;
		return getLPUsCountOfSpaceA_Sub(ppuCount, 
				*spaceALpu->aPartDims[1]->partitionDim, 
				partitionArgs[2]);
	}
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_A) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
		currentLpu->a = NULL;
		currentLpu->aPartDims = getaPartForSpaceALpu(
				spaceRootLpu->aPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->b = NULL;
		currentLpu->bPartDims = getbPartForSpaceALpu(
				spaceRootLpu->bPartDims, lpuCounts, nextLpuId, 
				partitionArgs[1]);
		currentLpu->c = NULL;
		currentLpu->cPartDims = getcPartForSpaceALpu(
				spaceRootLpu->cPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0], partitionArgs[1]);
		return currentLpu;
	}
	if (lpsId == Space_A_Sub) {
		SpaceA_LPU *spaceALpu = (SpaceA_LPU*) 
				lpsStates[Space_A]->lpu;
		SpaceA_Sub_LPU *currentLpu = new SpaceA_Sub_LPU;
		currentLpu->a = NULL;
		currentLpu->aPartDims = getaPartForSpaceA_SubLpu(
				spaceALpu->aPartDims, lpuCounts, nextLpuId, 
				partitionArgs[2]);
		currentLpu->b = NULL;
		currentLpu->bPartDims = getbPartForSpaceA_SubLpu(
				spaceALpu->bPartDims, lpuCounts, nextLpuId, 
				partitionArgs[2]);
		currentLpu->c = NULL;
		currentLpu->cPartDims = spaceALpu->cPartDims;
		return currentLpu;
	}
	return NULL;
}

