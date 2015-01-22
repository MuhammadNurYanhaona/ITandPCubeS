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

// to input-output
#include "../runtime/input_prompt.h"

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
		currentLpu->lpuId[0] = nextLpuId[0];
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

void s::initializeTask(ArrayMetadata arrayMetadata, 
		EnvironmentLinks environmentLinks, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		SPartition partition, 
		bool lower_triangular_system) {

	arrayMetadata.mDims[0] = environmentLinks.mDims[0];
	arrayMetadata.mDims[1] = environmentLinks.mDims[1];
	arrayMetadata.vDims[0] = environmentLinks.vDims[0];
	bool forwardDirection;
	forwardDirection = lower_triangular_system;
	if (forwardDirection) {
		taskGlobals.index_range = arrayMetadata.vDims[0].range;
	} else {
		taskGlobals.index_range.min = arrayMetadata.vDims[0].range.max;
		taskGlobals.index_range.max = 0;
	}
}

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void s::calculate_next_element(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, SPartition partition) {

	//-------------------- Local Copies of Metadata -----------------------------

	Dimension mPartDims[2];
	mPartDims[0] = *lpu.mPartDims[0]->partitionDim;
	mPartDims[1] = *lpu.mPartDims[1]->partitionDim;
	Dimension mStoreDims[2];
	mStoreDims[0] = *lpu.mPartDims[0]->storageDim;
	mStoreDims[1] = *lpu.mPartDims[1]->storageDim;
	Dimension vPartDims[1];
	vPartDims[0] = *lpu.vPartDims[0]->partitionDim;
	Dimension vStoreDims[1];
	vStoreDims[0] = *lpu.vPartDims[0]->storageDim;

	//----------------------- Computation Begins --------------------------------

	taskGlobals.v_element = lpu.v[threadLocals.k] / lpu.m[(threadLocals.k) * mStoreDims[1].length + threadLocals.k];
	lpu.v[threadLocals.k] = taskGlobals.v_element;
}

void s::column_sweep(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, SPartition partition) {

	//-------------------- Local Copies of Metadata -----------------------------

	Dimension mPartDims[2];
	mPartDims[0] = *lpu.mPartDims[0]->partitionDim;
	mPartDims[1] = *lpu.mPartDims[1]->partitionDim;
	Dimension mStoreDims[2];
	mStoreDims[0] = *lpu.mPartDims[0]->storageDim;
	mStoreDims[1] = *lpu.mPartDims[1]->storageDim;
	Dimension vPartDims[1];
	vPartDims[0] = *lpu.vPartDims[0]->partitionDim;
	Dimension vStoreDims[1];
	vStoreDims[0] = *lpu.vPartDims[0]->storageDim;

	//------------------- Local Variable Declarations ---------------------------

	float result;

	//----------------------- Computation Begins --------------------------------

	if (taskGlobals.index_range.min < taskGlobals.index_range.max) {
		{// scope entrance for parallel loop on index i
		int i;
		int iterationBound = mPartDims[0].range.max;
		int indexIncrement = 1;
		int indexMultiplier = 1;
		if (mPartDims[0].range.min > mPartDims[0].range.max) {
			iterationBound *= -1;
			indexIncrement *= -1;
			indexMultiplier = -1;
		}
		for (i = mPartDims[0].range.min; 
				indexMultiplier * i <= iterationBound; 
				i += indexIncrement) {
			int i_m_0 = i * mStoreDims[1].length;
			if (i > threadLocals.k) continue;
			result = "reduction";
		}
		}// scope exit for parallel loop on index i
	} else {
		{// scope entrance for parallel loop on index i
		int i;
		int iterationBound = mPartDims[0].range.max;
		int indexIncrement = 1;
		int indexMultiplier = 1;
		if (mPartDims[0].range.min > mPartDims[0].range.max) {
			iterationBound *= -1;
			indexIncrement *= -1;
			indexMultiplier = -1;
		}
		for (i = mPartDims[0].range.min; 
				indexMultiplier * i <= iterationBound; 
				i += indexIncrement) {
			int i_m_0 = i * mStoreDims[1].length;
			if (i < threadLocals.k) continue;
			result = "reduction";
		}
		}// scope exit for parallel loop on index i
	}
	lpu.v[threadLocals.k] = lpu.v[threadLocals.k] - result;
}

/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void s::run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		SPartition partition, ThreadStateImpl threadState) {

	// set the root LPU in the thread state so that calculation can start
	threadState.setRootLpu();

	{ // scope entrance for iterating LPUs of Space A
	int spaceALpuId = INVALID_ID;
	int spaceAIteration = 0;
	SpaceA_LPU *spaceALpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_A, Space_Root, spaceALpuId)) != NULL) {
		spaceALpu = (SpaceA_LPU*) lpu;
		{ // scope entrance for repeat loop
		int iterationBound = taskGlobals.index_range.max;
		int indexIncrement = 1;
		int indexMultiplier = 1;
		if (taskGlobals.index_range.min > taskGlobals.index_range.max) {
			iterationBound *= -1;
			indexIncrement *= -1;
			indexMultiplier = -1;
		}
		for (threadLocals.k = taskGlobals.index_range.min; 
				indexMultiplier * threadLocals.k <= iterationBound; 
				threadLocals.k += indexIncrement) {
			if (threadState.isValidPpu(Space_A)) {
				// invoking user computation
				calculate_next_element(*spaceALpu, 
						arrayMetadata,
						taskGlobals,
						threadLocals, partition);
			}
			if (threadState.isValidPpu(Space_A)) {
				// invoking user computation
				column_sweep(*spaceALpu, 
						arrayMetadata,
						taskGlobals,
						threadLocals, partition);
			}
		}
		} // scope exit for repeat loop
		spaceALpuId = spaceALpu->id;
		spaceAIteration++;
	}
	} // scope exit for iterating LPUs of Space A
}

/*-----------------------------------------------------------------------------------
main function
------------------------------------------------------------------------------------*/

int main() {

	std::cout << "Starting Substitution Task\n";

	// declaring common task related variables
	TaskGlobals taskGlobals;
	ThreadLocals threadLocals;
	EnvironmentLinks envLinks;
	ArrayMetadata metadata;
	SEnvironment environment;
	SPartition partition;

	// initializing variables that are environmental links 
	std::cout << "initializing environmental links\n";
	inprompt::readArrayDimensionInfo("m", 2, envLinks.mDims);
	inprompt::readArrayDimensionInfo("v", 1, envLinks.vDims);

	// determining values of partition parameters
	std::cout << "determining partition parameters\n";
	int *partitionArgs = NULL;

	// determining values of initialization parameters
	std::cout << "determining initialization parameters\n";
	bool lower_triangular_system;
	lower_triangular_system = readBoolean("lower_triangular_system");

	// invoking the initializer function
	std::cout << "invoking task initializer function\n";
	initializeTask(metadata, envLinks, taskGlobals, threadLocals, partition, lower_triangular_system);

	return 0;
}
