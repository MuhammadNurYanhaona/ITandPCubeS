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
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_A) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
		currentLpu->lpuId[0] = nextLpuId[0];
		currentLpu->lpuId[1] = nextLpuId[1];
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
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void mm::initializeTask(ArrayMetadata arrayMetadata, 
		EnvironmentLinks environmentLinks, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MMPartition partition) {

	arrayMetadata.aDims[0] = environmentLinks.aDims[0];
	arrayMetadata.aDims[1] = environmentLinks.aDims[1];
	arrayMetadata.bDims[0] = environmentLinks.bDims[0];
	arrayMetadata.bDims[1] = environmentLinks.bDims[1];
	arrayMetadata.cDims[0] = arrayMetadata.aDims[0];
	arrayMetadata.cDims[1] = arrayMetadata.bDims[1];
}

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void mm::mm_function0(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MMPartition partition) {

	//-------------------- Local Copies of Metadata -----------------------------

	Dimension aPartDims[2];
	aPartDims[0] = *lpu.aPartDims[0]->partitionDim;
	aPartDims[1] = *lpu.aPartDims[1]->partitionDim;
	Dimension aStoreDims[2];
	aStoreDims[0] = *lpu.aPartDims[0]->storageDim;
	aStoreDims[1] = *lpu.aPartDims[1]->storageDim;
	Dimension bPartDims[2];
	bPartDims[0] = *lpu.bPartDims[0]->partitionDim;
	bPartDims[1] = *lpu.bPartDims[1]->partitionDim;
	Dimension bStoreDims[2];
	bStoreDims[0] = *lpu.bPartDims[0]->storageDim;
	bStoreDims[1] = *lpu.bPartDims[1]->storageDim;
	Dimension cPartDims[2];
	cPartDims[0] = *lpu.cPartDims[0]->partitionDim;
	cPartDims[1] = *lpu.cPartDims[1]->partitionDim;
	Dimension cStoreDims[2];
	cStoreDims[0] = *lpu.cPartDims[0]->storageDim;
	cStoreDims[1] = *lpu.cPartDims[1]->storageDim;

	//----------------------- Computation Begins --------------------------------

	{// scope entrance for parallel loop on index i
	int i;
	int iterationBound = cPartDims[0].range.max;
	int indexIncrement = 1;
	int indexMultiplier = 1;
	if (cPartDims[0].range.min > cPartDims[0].range.max) {
		iterationBound *= -1;
		indexIncrement *= -1;
		indexMultiplier = -1;
	}
	for (i = cPartDims[0].range.min; 
			indexMultiplier * i <= iterationBound; 
			i += indexIncrement) {
		int i_c_0 = i * cStoreDims[1].length;
		int i_a_0 = i * aStoreDims[1].length;
		{// scope entrance for parallel loop on index j
		int j;
		int iterationBound = cPartDims[1].range.max;
		int indexIncrement = 1;
		int indexMultiplier = 1;
		if (cPartDims[1].range.min > cPartDims[1].range.max) {
			iterationBound *= -1;
			indexIncrement *= -1;
			indexMultiplier = -1;
		}
		for (j = cPartDims[1].range.min; 
				indexMultiplier * j <= iterationBound; 
				j += indexIncrement) {
			{// scope entrance for parallel loop on index k
			int k;
			int iterationBound = aPartDims[1].range.max;
			int indexIncrement = 1;
			int indexMultiplier = 1;
			if (aPartDims[1].range.min > aPartDims[1].range.max) {
				iterationBound *= -1;
				indexIncrement *= -1;
				indexMultiplier = -1;
			}
			for (k = aPartDims[1].range.min; 
					indexMultiplier * k <= iterationBound; 
					k += indexIncrement) {
				int k_b_0 = k * bStoreDims[1].length;
				lpu.c[i_c_0 + j] = lpu.c[i_c_0 + j] + lpu.a[i_a_0 + k] * lpu.b[k_b_0 + j];
			}
			}// scope exit for parallel loop on index k
		}
		}// scope exit for parallel loop on index j
	}
	}// scope exit for parallel loop on index i
}

/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void mm::run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MMPartition partition, ThreadStateImpl threadState) {

	// set the root LPU in the thread state so that calculation can start
	threadState.setRootLpu();

	{ // scope entrance for iterating LPUs of Space A
	int spaceALpuId = INVALID_ID;
	int spaceAIteration = 0;
	SpaceA_LPU *spaceALpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_A, Space_Root, spaceALpuId)) != NULL) {
		spaceALpu = (SpaceA_LPU*) lpu;
		if (threadState.isValidPpu(Space_A)) {
			// invoking user computation
			mm_function0(*spaceALpu, 
					arrayMetadata,
					taskGlobals,
					threadLocals, partition);
		}
		spaceALpuId = spaceALpu->id;
		spaceAIteration++;
	}
	} // scope exit for iterating LPUs of Space A
}

/*-----------------------------------------------------------------------------------
main function
------------------------------------------------------------------------------------*/

int main() {

	std::cout << "Starting Matrix Multiply Task\n";

	// declaring common task related variables
	TaskGlobals taskGlobals;
	ThreadLocals threadLocals;
	EnvironmentLinks envLinks;
	ArrayMetadata metadata;
	MMEnvironment environment;
	MMPartition partition;

	// initializing variables that are environmental links 
	std::cout << "initializing environmental links\n";
	inprompt::readArrayDimensionInfo("a", 2, envLinks.aDims);
	inprompt::readArrayDimensionInfo("b", 2, envLinks.bDims);

	// determining values of partition parameters
	std::cout << "determining partition parameters\n";
	int *partitionArgs = NULL;
	partitionArgs = new int[2];
	partition.k = inprompt::readPrimitive <int> ("k");
	partitionArgs[0] = partition.k;
	partition.l = inprompt::readPrimitive <int> ("l");
	partitionArgs[1] = partition.l;

	// determining values of initialization parameters
	std::cout << "determining initialization parameters\n";

	// invoking the initializer function
	std::cout << "invoking task initializer function\n";
	initializeTask(metadata, envLinks, taskGlobals, threadLocals, partition);

	return 0;
}
