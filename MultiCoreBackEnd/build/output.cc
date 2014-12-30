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

/*-----------------------------------------------------------------------------------
macro definitions for LPSes
------------------------------------------------------------------------------------*/
#define Space_Root 0
#define Space_A 1
#define Space_B 2
#define Space_C 3
#define Space_D 4
#define Space_Count 5

/*-----------------------------------------------------------------------------------
macro definitions for PPS counts
------------------------------------------------------------------------------------*/
#define Space_5_PPUs 1
#define Space_4_Par_5_PPUs 2
#define Space_3_Par_4_PPUs 4
#define Space_2_Par_3_PPUs 2
#define Space_1_Par_2_PPUs 4

/*-----------------------------------------------------------------------------------
macro definitions for total and par core thread counts
------------------------------------------------------------------------------------*/
#define Total_Threads 32
#define Threads_Par_Core 4

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *getLPUsCountOfSpaceB(int ppuCount, Dimension aDim2) {
	int *count = new int[1];
	count[0] = stride_partitionCount(aDim2, ppuCount);
	return count;
}

int *getLPUsCountOfSpaceC(int ppuCount, Dimension uDim2) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(uDim2, ppuCount, 1);
	return count;
}

int *getLPUsCountOfSpaceD(int ppuCount, Dimension uDim1, int s) {
	int *count = new int[1];
	count[0] = block_size_partitionCount(uDim1, ppuCount, s);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **getaPartForSpaceBLpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **aLpuDims = new PartitionDimension*[2];
	aLpuDims[0] = aParentLpuDims[0];
	aLpuDims[1] = new PartitionDimension;
	aLpuDims[1]->storageDim = aParentLpuDims[1]->partitionDim;
	aLpuDims[1]->partitionDim = stride_getRange(*aParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return aLpuDims;
}

PartitionDimension **getlPartForSpaceBLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = lParentLpuDims[0];
	lLpuDims[1] = new PartitionDimension;
	lLpuDims[1]->storageDim = lParentLpuDims[1]->partitionDim;
	lLpuDims[1]->partitionDim = stride_getRange(*lParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return lLpuDims;
}

PartitionDimension **getuPartForSpaceBLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **uLpuDims = new PartitionDimension*[2];
	uLpuDims[0] = uParentLpuDims[0];
	uLpuDims[1] = new PartitionDimension;
	uLpuDims[1]->storageDim = uParentLpuDims[1]->partitionDim;
	uLpuDims[1]->partitionDim = stride_getRange(*uParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0]);
	return uLpuDims;
}

PartitionDimension **getlPartForSpaceCLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = lParentLpuDims[0];
	lLpuDims[1] = new PartitionDimension;
	lLpuDims[1]->storageDim = lParentLpuDims[1]->partitionDim;
	lLpuDims[1]->partitionDim = block_size_getRange(*lParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	return lLpuDims;
}

PartitionDimension **getuPartForSpaceCLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **uLpuDims = new PartitionDimension*[2];
	uLpuDims[0] = uParentLpuDims[0];
	uLpuDims[1] = new PartitionDimension;
	uLpuDims[1]->storageDim = uParentLpuDims[1]->partitionDim;
	uLpuDims[1]->partitionDim = block_size_getRange(*uParentLpuDims[1]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	return uLpuDims;
}

PartitionDimension **getlPartForSpaceDLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId, int s) {
	PartitionDimension **lLpuDims = new PartitionDimension*[2];
	lLpuDims[0] = new PartitionDimension;
	lLpuDims[0]->storageDim = lParentLpuDims[0]->partitionDim;
	lLpuDims[0]->partitionDim = block_size_getRange(*lParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], s, 0, 0);
	lLpuDims[1] = lParentLpuDims[1];
	return lLpuDims;
}

PartitionDimension **getuPartForSpaceDLpu(PartitionDimension **uParentLpuDims, 
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
Data structures representing LPS and LPU contents 
------------------------------------------------------------------------------------*/

class SpaceRoot_Content {
  public:
	float *a;
	float *l;
	float *l_column;
	int *p;
	float *u;
};

class SpaceRoot_LPU : public LPU {
  public:
	float *a;
	PartitionDimension **aPartDims;
	float *l;
	PartitionDimension **lPartDims;
	float *l_column;
	PartitionDimension **l_columnPartDims;
	int *p;
	PartitionDimension **pPartDims;
	float *u;
	PartitionDimension **uPartDims;
};

class SpaceA_Content {
  public:
	int *p;
};

class SpaceA_LPU : public LPU {
  public:
	int *p;
	PartitionDimension **pPartDims;
};

class SpaceB_Content {
  public:
	float *a;
	float *l;
	float *l_column;
	float *u;
};

class SpaceB_LPU : public LPU {
  public:
	float *a;
	PartitionDimension **aPartDims;
	float *l;
	PartitionDimension **lPartDims;
	float *l_column;
	PartitionDimension **l_columnPartDims;
	float *u;
	PartitionDimension **uPartDims;
};

class SpaceC_Content {
  public:
	float *l;
	float *l_column;
	float *u;
};

class SpaceC_LPU : public LPU {
  public:
	float *l;
	PartitionDimension **lPartDims;
	float *l_column;
	PartitionDimension **l_columnPartDims;
	float *u;
	PartitionDimension **uPartDims;
};

class SpaceD_Content {
  public:
	float *l;
	float *u;
};

class SpaceD_LPU : public LPU {
  public:
	float *l;
	PartitionDimension **lPartDims;
	float *u;
	PartitionDimension **uPartDims;
};

/*-----------------------------------------------------------------------------------
Data structures for Array-Metadata and Environment-Links 
------------------------------------------------------------------------------------*/

class ArrayMetadata {
  public:
	Dimension aDims[2];
	Dimension lDims[2];
	Dimension l_columnDims[1];
	Dimension pDims[1];
	Dimension uDims[2];
};

class EnvironmentLinks {
  public:
	float *a;
	Dimension aDims[2];
};

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	int threadCount;
	int ppuCount;
	int groupThreadId;

	// for Space A;
	threadCount = Total_Threads;
	ppuCount = threadCount;
	groupThreadId = idsArray[Space_Root] % ppuCount;
	threadIds->ppuIds[Space_A].groupId = idsArray[Space_Root] / ppuCount;
	threadIds->ppuIds[Space_A].ppuCount = ppuCount;
	if (groupThreadId == 0) threadIds->ppuIds[Space_A].id
			= threadIds->ppuIds[Space_A].groupId;
	else threadIds->ppuIds[Space_A].id = INVALID_ID;
	idsArray[Space_A] = groupThreadId;

	// for Space B;
	threadCount = threadIds->ppuIds[Space_A].ppuCount;
	ppuCount = threadCount / 32;
	groupThreadId = idsArray[Space_A] % ppuCount;
	threadIds->ppuIds[Space_B].groupId = idsArray[Space_A] / ppuCount;
	threadIds->ppuIds[Space_B].ppuCount = ppuCount;
	if (groupThreadId == 0) threadIds->ppuIds[Space_B].id
			= threadIds->ppuIds[Space_B].groupId;
	else threadIds->ppuIds[Space_B].id = INVALID_ID;
	idsArray[Space_B] = groupThreadId;

	// for Space C;
	threadCount = threadIds->ppuIds[Space_A].ppuCount;
	ppuCount = threadCount / 4;
	groupThreadId = idsArray[Space_A] % ppuCount;
	threadIds->ppuIds[Space_C].groupId = idsArray[Space_A] / ppuCount;
	threadIds->ppuIds[Space_C].ppuCount = ppuCount;
	if (groupThreadId == 0) threadIds->ppuIds[Space_C].id
			= threadIds->ppuIds[Space_C].groupId;
	else threadIds->ppuIds[Space_C].id = INVALID_ID;
	idsArray[Space_C] = groupThreadId;

	// for Space D;
	threadCount = threadIds->ppuIds[Space_C].ppuCount;
	ppuCount = threadCount / 8;
	groupThreadId = idsArray[Space_C] % ppuCount;
	threadIds->ppuIds[Space_D].groupId = idsArray[Space_C] / ppuCount;
	threadIds->ppuIds[Space_D].ppuCount = ppuCount;
	if (groupThreadId == 0) threadIds->ppuIds[Space_D].id
			= threadIds->ppuIds[Space_D].groupId;
	else threadIds->ppuIds[Space_D].id = INVALID_ID;
	idsArray[Space_D] = groupThreadId;

	return threadIds;
}

/*-----------------------------------------------------------------------------------
Thread-State implementation class for the task
------------------------------------------------------------------------------------*/

class ThreadStateImpl : public ThreadState {
  public:
	void setLpsParentIndexMap();
        virtual void setRootLpu() = 0;
        int *computeLpuCounts(int lpsId);
        LPU *computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId);
};

// Construction of task specific LPS hierarchy index map
void ThreadStateImpl::setLpsParentIndexMap() {
	lpsParentIndexMap = new int[Space_Count];
	lpsParentIndexMap[Space_Root] = INVALID_ID;
	lpsParentIndexMap[Space_A] = Space_Root;
	lpsParentIndexMap[Space_B] = Space_A;
	lpsParentIndexMap[Space_C] = Space_A;
	lpsParentIndexMap[Space_D] = Space_C;
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

