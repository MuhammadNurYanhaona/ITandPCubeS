/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "lu_factorization.h"

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

using namespace luf;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *luf::getLPUsCountOfSpaceB(int ppuCount, Dimension aDim2) {
	int *count = new int[1];
	count[0] = stride_partitionCount(aDim2, ppuCount);
	return count;
}

int *luf::getLPUsCountOfSpaceC(int ppuCount, Dimension uDim2) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(uDim2, ppuCount, 1);
	return count;
}

int *luf::getLPUsCountOfSpaceD(int ppuCount, Dimension uDim1, int s) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(uDim1, ppuCount, s);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **luf::getaPartForSpaceBLpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **aLpuDims = new PartitionDimension*[2];
	aLpuDims[0] = aParentLpuDims[0];
	aLpuDims[1] = new PartitionDimension;
	aLpuDims[1]->storageDim = aParentLpuDims[1]->partitionDim;
	aLpuDims[1]->partitionDim = stride_getRange(*aParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return aLpuDims;
}

PartitionDimension **luf::getlPartForSpaceBLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = lParentLpuDims[0];
	lLpuDims[1] = new PartitionDimension;
	lLpuDims[1]->storageDim = lParentLpuDims[1]->partitionDim;
	lLpuDims[1]->partitionDim = stride_getRange(*lParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return lLpuDims;
}

PartitionDimension **luf::getuPartForSpaceBLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **uLpuDims = new PartitionDimension*[2];
	uLpuDims[0] = uParentLpuDims[0];
	uLpuDims[1] = new PartitionDimension;
	uLpuDims[1]->storageDim = uParentLpuDims[1]->partitionDim;
	uLpuDims[1]->partitionDim = stride_getRange(*uParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return uLpuDims;
}

PartitionDimension **luf::getlPartForSpaceCLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = lParentLpuDims[0];
	lLpuDims[1] = new PartitionDimension;
	lLpuDims[1]->storageDim = lParentLpuDims[1]->partitionDim;
	lLpuDims[1]->partitionDim = block_size_getRange(*lParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	return lLpuDims;
}

PartitionDimension **luf::getuPartForSpaceCLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **uLpuDims = new PartitionDimension*[2];
	uLpuDims[0] = uParentLpuDims[0];
	uLpuDims[1] = new PartitionDimension;
	uLpuDims[1]->storageDim = uParentLpuDims[1]->partitionDim;
	uLpuDims[1]->partitionDim = block_size_getRange(*uParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	return uLpuDims;
}

PartitionDimension **luf::getlPartForSpaceDLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId, int s) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = new PartitionDimension;
	lLpuDims[0]->storageDim = lParentLpuDims[0]->partitionDim;
	lLpuDims[0]->partitionDim = block_size_getRange(*lParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], s, 0, 0);
	lLpuDims[1] = lParentLpuDims[1];
	return lLpuDims;
}

PartitionDimension **luf::getuPartForSpaceDLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId, int s) {
	PartitionDimension **uLpuDims = new PartitionDimension*[2];
	uLpuDims[0] = new PartitionDimension;
	uLpuDims[0]->storageDim = uParentLpuDims[0]->partitionDim;
	uLpuDims[0]->partitionDim = block_size_getRange(*uParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], s, 0, 0);
	uLpuDims[1] = uParentLpuDims[1];
	return uLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *luf::getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	int threadCount;
	int groupSize;
	int groupThreadId;

	// for Space A;
	threadCount = Total_Threads;
	groupSize = threadCount;
	groupThreadId = idsArray[Space_Root] % groupSize;
	threadIds->ppuIds[Space_A].groupId = idsArray[Space_Root] / groupSize;
	threadIds->ppuIds[Space_A].ppuCount = 2;
	threadIds->ppuIds[Space_A].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_A].id
			= threadIds->ppuIds[Space_A].groupId;
	else threadIds->ppuIds[Space_A].id = INVALID_ID;
	idsArray[Space_A] = groupThreadId;

	// for Space B;
	threadCount = threadIds->ppuIds[Space_A].groupSize;
	groupSize = threadCount / 32;
	groupThreadId = idsArray[Space_A] % groupSize;
	threadIds->ppuIds[Space_B].groupId = idsArray[Space_A] / groupSize;
	threadIds->ppuIds[Space_B].ppuCount = 32;
	threadIds->ppuIds[Space_B].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_B].id
			= threadIds->ppuIds[Space_B].groupId;
	else threadIds->ppuIds[Space_B].id = INVALID_ID;
	idsArray[Space_B] = groupThreadId;

	// for Space C;
	threadCount = threadIds->ppuIds[Space_A].groupSize;
	groupSize = threadCount / 4;
	groupThreadId = idsArray[Space_A] % groupSize;
	threadIds->ppuIds[Space_C].groupId = idsArray[Space_A] / groupSize;
	threadIds->ppuIds[Space_C].ppuCount = 4;
	threadIds->ppuIds[Space_C].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_C].id
			= threadIds->ppuIds[Space_C].groupId;
	else threadIds->ppuIds[Space_C].id = INVALID_ID;
	idsArray[Space_C] = groupThreadId;

	// for Space D;
	threadCount = threadIds->ppuIds[Space_C].groupSize;
	groupSize = threadCount / 8;
	groupThreadId = idsArray[Space_C] % groupSize;
	threadIds->ppuIds[Space_D].groupId = idsArray[Space_C] / groupSize;
	threadIds->ppuIds[Space_D].ppuCount = 8;
	threadIds->ppuIds[Space_D].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_D].id
			= threadIds->ppuIds[Space_D].groupId;
	else threadIds->ppuIds[Space_D].id = INVALID_ID;
	idsArray[Space_D] = groupThreadId;

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
	lpsParentIndexMap[Space_B] = Space_A;
	lpsParentIndexMap[Space_C] = Space_A;
	lpsParentIndexMap[Space_D] = Space_C;
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

	lpu->l = NULL;
	lpu->lPartDims = new PartitionDimension*[2];
	lpu->lPartDims[0] = new PartitionDimension;
	lpu->lPartDims[0]->storageDim = lpu->lPartDims[0]->partitionDim
			= &arrayMetadata.lDims[0];
	lpu->lPartDims[1] = new PartitionDimension;
	lpu->lPartDims[1]->storageDim = lpu->lPartDims[1]->partitionDim
			= &arrayMetadata.lDims[1];

	lpu->l_column = NULL;
	lpu->l_columnPartDims = new PartitionDimension*[1];
	lpu->l_columnPartDims[0] = new PartitionDimension;
	lpu->l_columnPartDims[0]->storageDim = lpu->l_columnPartDims[0]->partitionDim
			= &arrayMetadata.l_columnDims[0];

	lpu->p = NULL;
	lpu->pPartDims = new PartitionDimension*[1];
	lpu->pPartDims[0] = new PartitionDimension;
	lpu->pPartDims[0]->storageDim = lpu->pPartDims[0]->partitionDim
			= &arrayMetadata.pDims[0];

	lpu->u = NULL;
	lpu->uPartDims = new PartitionDimension*[2];
	lpu->uPartDims[0] = new PartitionDimension;
	lpu->uPartDims[0]->storageDim = lpu->uPartDims[0]->partitionDim
			= &arrayMetadata.uDims[0];
	lpu->uPartDims[1] = new PartitionDimension;
	lpu->uPartDims[1]->storageDim = lpu->uPartDims[1]->partitionDim
			= &arrayMetadata.uDims[1];

	lpsStates[Space_Root]->lpu = lpu;
}

// Implementation of task specific compute-LPU-Count function 
int *ThreadStateImpl::computeLpuCounts(int lpsId) {
	if (lpsId == Space_Root) {
		return NULL;
	}
	if (lpsId == Space_A) {
		return NULL;
	}
	if (lpsId == Space_B) {
		int ppuCount = threadIds->ppuIds[Space_B].ppuCount;
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceB(ppuCount, 
				*spaceRootLpu->aPartDims[1]->partitionDim);
	}
	if (lpsId == Space_C) {
		int ppuCount = threadIds->ppuIds[Space_C].ppuCount;
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceC(ppuCount, 
				*spaceRootLpu->uPartDims[1]->partitionDim);
	}
	if (lpsId == Space_D) {
		int ppuCount = threadIds->ppuIds[Space_D].ppuCount;
		SpaceC_LPU *spaceCLpu = (SpaceC_LPU*) 
				lpsStates[Space_C]->lpu;
		return getLPUsCountOfSpaceD(ppuCount, 
				*spaceCLpu->uPartDims[0]->partitionDim, 
				partitionArgs[0]);
	}
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_A) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
		currentLpu->p = NULL;
		currentLpu->pPartDims = spaceRootLpu->pPartDims;
		return currentLpu;
	}
	if (lpsId == Space_B) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceB_LPU *currentLpu = new SpaceB_LPU;
		currentLpu->a = NULL;
		currentLpu->aPartDims = getaPartForSpaceBLpu(
				spaceRootLpu->aPartDims, lpuCounts, nextLpuId);
		currentLpu->l = NULL;
		currentLpu->lPartDims = getlPartForSpaceBLpu(
				spaceRootLpu->lPartDims, lpuCounts, nextLpuId);
		currentLpu->l_column = NULL;
		currentLpu->l_columnPartDims = spaceRootLpu->l_columnPartDims;
		currentLpu->u = NULL;
		currentLpu->uPartDims = getuPartForSpaceBLpu(
				spaceRootLpu->uPartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	if (lpsId == Space_C) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceC_LPU *currentLpu = new SpaceC_LPU;
		currentLpu->l = NULL;
		currentLpu->lPartDims = getlPartForSpaceCLpu(
				spaceRootLpu->lPartDims, lpuCounts, nextLpuId);
		currentLpu->l_column = NULL;
		currentLpu->l_columnPartDims = spaceRootLpu->l_columnPartDims;
		currentLpu->u = NULL;
		currentLpu->uPartDims = getuPartForSpaceCLpu(
				spaceRootLpu->uPartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	if (lpsId == Space_D) {
		SpaceC_LPU *spaceCLpu = (SpaceC_LPU*) 
				lpsStates[Space_C]->lpu;
		SpaceD_LPU *currentLpu = new SpaceD_LPU;
		currentLpu->l = NULL;
		currentLpu->lPartDims = getlPartForSpaceDLpu(
				spaceCLpu->lPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->u = NULL;
		currentLpu->uPartDims = getuPartForSpaceDLpu(
				spaceCLpu->uPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		return currentLpu;
	}
	return NULL;
}

