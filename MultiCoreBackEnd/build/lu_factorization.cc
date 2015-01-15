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
#include <stdio.h>

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

using namespace luf;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *luf::getLPUsCountOfSpaceB(int ppuCount, Dimension aDim2) {
	int *count = new int[1];
	count[0] = stride_partitionCount(aDim2, ppuCount);
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
	threadIds->ppuIds[Space_A].ppuCount = 1;
	threadIds->ppuIds[Space_A].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_A].id
			= threadIds->ppuIds[Space_A].groupId;
	else threadIds->ppuIds[Space_A].id = INVALID_ID;
	idsArray[Space_A] = groupThreadId;

	// for Space B;
	threadCount = threadIds->ppuIds[Space_A].groupSize;
	groupSize = threadCount / 16;
	groupThreadId = idsArray[Space_A] % groupSize;
	threadIds->ppuIds[Space_B].groupId = idsArray[Space_A] / groupSize;
	threadIds->ppuIds[Space_B].ppuCount = 16;
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
	lpsParentIndexMap[Space_B] = Space_A;
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
		currentLpu->lpuId[0] = nextLpuId[0];
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
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void luf::initializeTask(TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		LUFPartition partition) {

	arrayMetadata.aDims[0] = environmentLinks.aDims[0];
	arrayMetadata.aDims[1] = environmentLinks.aDims[1];
	arrayMetadata.lDims[0] = arrayMetadata.aDims[0];
	arrayMetadata.lDims[1] = arrayMetadata.aDims[1];
	arrayMetadata.uDims[0] = arrayMetadata.lDims[0];
	arrayMetadata.uDims[1] = arrayMetadata.lDims[1];
	arrayMetadata.l_columnDims[0] = arrayMetadata.lDims[0];
	arrayMetadata.pDims[0] = arrayMetadata.aDims[0];
}

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void luf::Prepare(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->storageDim;
	aPartDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension lPartDims[2];
	lPartDims[0] = *lpu.lPartDims[0]->storageDim;
	lPartDims[1] = *lpu.lPartDims[1]->storageDim;
	Dimension l_columnPartDims[1];
	l_columnPartDims[0] = *lpu.l_columnPartDims[0]->storageDim;
	Dimension uPartDims[2];
	uPartDims[0] = *lpu.uPartDims[0]->storageDim;
	uPartDims[1] = *lpu.uPartDims[1]->storageDim;
}

void luf::Select_Pivot(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->storageDim;
	aPartDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension lPartDims[2];
	lPartDims[0] = *lpu.lPartDims[0]->storageDim;
	lPartDims[1] = *lpu.lPartDims[1]->storageDim;
	Dimension l_columnPartDims[1];
	l_columnPartDims[0] = *lpu.l_columnPartDims[0]->storageDim;
	Dimension uPartDims[2];
	uPartDims[0] = *lpu.uPartDims[0]->storageDim;
	uPartDims[1] = *lpu.uPartDims[1]->storageDim;
}

void luf::Store_Pivot(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension pPartDims[1];
	pPartDims[0] = *lpu.pPartDims[0]->storageDim;
}

void luf::Interchange_Rows(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->storageDim;
	aPartDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension lPartDims[2];
	lPartDims[0] = *lpu.lPartDims[0]->storageDim;
	lPartDims[1] = *lpu.lPartDims[1]->storageDim;
	Dimension l_columnPartDims[1];
	l_columnPartDims[0] = *lpu.l_columnPartDims[0]->storageDim;
	Dimension uPartDims[2];
	uPartDims[0] = *lpu.uPartDims[0]->storageDim;
	uPartDims[1] = *lpu.uPartDims[1]->storageDim;
}

void luf::Update_Lower(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->storageDim;
	aPartDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension lPartDims[2];
	lPartDims[0] = *lpu.lPartDims[0]->storageDim;
	lPartDims[1] = *lpu.lPartDims[1]->storageDim;
	Dimension l_columnPartDims[1];
	l_columnPartDims[0] = *lpu.l_columnPartDims[0]->storageDim;
	Dimension uPartDims[2];
	uPartDims[0] = *lpu.uPartDims[0]->storageDim;
	uPartDims[1] = *lpu.uPartDims[1]->storageDim;
}

void luf::Update_Upper(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition) {

	//create local variables for array dimensions 
	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->storageDim;
	aPartDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension lPartDims[2];
	lPartDims[0] = *lpu.lPartDims[0]->storageDim;
	lPartDims[1] = *lpu.lPartDims[1]->storageDim;
	Dimension l_columnPartDims[1];
	l_columnPartDims[0] = *lpu.l_columnPartDims[0]->storageDim;
	Dimension uPartDims[2];
	uPartDims[0] = *lpu.uPartDims[0]->storageDim;
	uPartDims[1] = *lpu.uPartDims[1]->storageDim;
}

/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void luf::run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		LUFPartition partition, ThreadStateImpl threadState) {

	// set the root LPU in the thread state so that calculation can start
	threadState.setRootLpu();

	{ // scope entrance for iterating LPUs of Space B
	int spaceBLpuId = INVALID_ID;
	int spaceBIteration = 0;
	SpaceB_LPU *spaceBLpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_B, Space_Root, spaceBLpuId)) != NULL) {
		spaceBLpu = (SpaceB_LPU*) lpu;
		if (threadState.isValidPpu(Space_B)) {
			// invoking user computation
			Prepare(*spaceBLpu, 
					arrayMetadata,
					taskGlobals,
					threadLocals, partition);
		}
		spaceBLpuId = spaceBLpu->id;
		spaceBIteration++;
	}
	} // scope exit for iterating LPUs of Space B
	{ // scope entrance for iterating LPUs of Space A
	int spaceALpuId = INVALID_ID;
	int spaceAIteration = 0;
	SpaceA_LPU *spaceALpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_A, Space_Root, spaceALpuId)) != NULL) {
		spaceALpu = (SpaceA_LPU*) lpu;
		{ // scope entrance for repeat loop
		int iterationBound = arrayMetadata.aDims[0].range.max;
		int indexIncrement = 1;
		int indexMultiplier = 1;
		if (arrayMetadata.aDims[0].range.min > arrayMetadata.aDims[0].range.max) {
			iterationBound *= -1;
			indexIncrement *= -1;
			indexMultiplier = -1;
		}
		for (threadLocals.k = arrayMetadata.aDims[0].range.min; 
				indexMultiplier * threadLocals.k <= iterationBound; 
				threadLocals.k += indexIncrement) {
			{ // scope entrance for iterating LPUs of Space B
			int spaceBLpuId = INVALID_ID;
			int spaceBIteration = 0;
			SpaceB_LPU *spaceBLpu = NULL;
			LPU *lpu = NULL;
			while((lpu = threadState.getNextLpu(Space_B, Space_A, spaceBLpuId)) != NULL) {
				spaceBLpu = (SpaceB_LPU*) lpu;
				// invoking user computation
				Select_Pivot(*spaceBLpu, 
						arrayMetadata,
						taskGlobals,
						threadLocals, partition);
				spaceBLpuId = spaceBLpu->id;
				spaceBIteration++;
			}
			threadState.removeIterationBound(Space_A);
			} // scope exit for iterating LPUs of Space B
			if (threadState.isValidPpu(Space_A)) {
				// invoking user computation
				Store_Pivot(*spaceALpu, 
						arrayMetadata,
						taskGlobals,
						threadLocals, partition);
			}
			{ // scope entrance for iterating LPUs of Space B
			int spaceBLpuId = INVALID_ID;
			int spaceBIteration = 0;
			SpaceB_LPU *spaceBLpu = NULL;
			LPU *lpu = NULL;
			while((lpu = threadState.getNextLpu(Space_B, Space_A, spaceBLpuId)) != NULL) {
				spaceBLpu = (SpaceB_LPU*) lpu;
				if (threadState.isValidPpu(Space_B)) {
					// invoking user computation
					Interchange_Rows(*spaceBLpu, 
							arrayMetadata,
							taskGlobals,
							threadLocals, partition);
				}
				if (threadState.isValidPpu(Space_B)) {
					// invoking user computation
					Update_Lower(*spaceBLpu, 
							arrayMetadata,
							taskGlobals,
							threadLocals, partition);
				}
				if (threadState.isValidPpu(Space_B)) {
					// invoking user computation
					Update_Upper(*spaceBLpu, 
							arrayMetadata,
							taskGlobals,
							threadLocals, partition);
				}
				spaceBLpuId = spaceBLpu->id;
				spaceBIteration++;
			}
			threadState.removeIterationBound(Space_A);
			} // scope exit for iterating LPUs of Space B
		}
		} // scope exit for repeat loop
		spaceALpuId = spaceALpu->id;
		spaceAIteration++;
	}
	} // scope exit for iterating LPUs of Space A
}

