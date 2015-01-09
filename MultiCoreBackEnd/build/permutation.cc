/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "permutation.h"

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

using namespace p;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *p::getLPUsCountOfSpaceA(int ppuCount, Dimension pDim1, int n) {
	int *count = new int[1];
	count[0] = block_count_partitionCount(pDim1, ppuCount, n);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **p::getpPartForSpaceALpu(PartitionDimension **pParentLpuDims, 
		int *lpuCount, int *lpuId, int n) {
	PartitionDimension **pLpuDims = new PartitionDimension*[1];
	pLpuDims[0] = new PartitionDimension;
	pLpuDims[0]->storageDim = pParentLpuDims[0]->partitionDim;
	pLpuDims[0]->partitionDim = block_count_getRange(*pParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], n, 0, 0);
	return pLpuDims;
}

PartitionDimension **p::getuPartForSpaceALpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId, int n) {
	PartitionDimension **uLpuDims = new PartitionDimension*[1];
	uLpuDims[0] = new PartitionDimension;
	uLpuDims[0]->storageDim = uParentLpuDims[0]->partitionDim;
	uLpuDims[0]->partitionDim = block_count_getRange(*uParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], n, 0, 0);
	return uLpuDims;
}

PartitionDimension **p::getvPartForSpaceALpu(PartitionDimension **vParentLpuDims, 
		int *lpuCount, int *lpuId, int n) {
	PartitionDimension **vLpuDims = new PartitionDimension*[1];
	vLpuDims[0] = new PartitionDimension;
	vLpuDims[0]->storageDim = vParentLpuDims[0]->partitionDim;
	vLpuDims[0]->partitionDim = block_count_getRange(*vParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], n, 0, 0);
	return vLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *p::getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	int threadCount;
	int groupSize;
	int groupThreadId;

	// for Space A;
	threadCount = Total_Threads;
	groupSize = threadCount / 1;
	groupThreadId = idsArray[Space_Root] % groupSize;
	threadIds->ppuIds[Space_A].groupId = idsArray[Space_Root] / groupSize;
	threadIds->ppuIds[Space_A].ppuCount = 1;
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
	lpu->p = NULL;
	lpu->pPartDims = new PartitionDimension*[1];
	lpu->pPartDims[0] = new PartitionDimension;
	lpu->pPartDims[0]->storageDim = lpu->pPartDims[0]->partitionDim
			= &arrayMetadata.pDims[0];

	lpu->u = NULL;
	lpu->uPartDims = new PartitionDimension*[1];
	lpu->uPartDims[0] = new PartitionDimension;
	lpu->uPartDims[0]->storageDim = lpu->uPartDims[0]->partitionDim
			= &arrayMetadata.uDims[0];

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
				*spaceRootLpu->pPartDims[0]->partitionDim, 
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
		currentLpu->pPartDims = getpPartForSpaceALpu(
				spaceRootLpu->pPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->u = NULL;
		currentLpu->uPartDims = getuPartForSpaceALpu(
				spaceRootLpu->uPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->v = NULL;
		currentLpu->vPartDims = getvPartForSpaceALpu(
				spaceRootLpu->vPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		return currentLpu;
	}
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void p::initializeTask(TaskGlobals taskGlobals, ThreadLocals threadLocals) {
	arrayMetadata.pDims[0] = environmentLinks.pDims[0];
	arrayMetadata.uDims[0] = environmentLinks.uDims[0];
}

