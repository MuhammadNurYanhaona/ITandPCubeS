/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "matrix_multiply.h"

/*-----------------------------------------------------------------------------------
header files included for different purposes
------------------------------------------------------------------------------------*/
// for error reporting and diagnostics
#include <iostream>
#include <fstream>
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

// to input-output and initialization
#include "../runtime/input_prompt.h"
#include "../runtime/allocator.h"


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

void mm::getaPartForSpaceALpu(PartDimension *aLpuDims, 
		PartDimension *aParentLpuDims, 
		int *lpuCount, int *lpuId, int k) {
	aLpuDims[0].storage = aParentLpuDims[0].storage;
	aLpuDims[0].partition = block_size_getRange(aParentLpuDims[0].partition, 
			lpuCount[0], lpuId[0], k, 0, 0);
	aLpuDims[1] = aParentLpuDims[1];
}

void mm::getbPartForSpaceALpu(PartDimension *bLpuDims, 
		PartDimension *bParentLpuDims, 
		int *lpuCount, int *lpuId, int l) {
	bLpuDims[0].storage = bParentLpuDims[0].storage;
	bLpuDims[0].partition = block_size_getRange(bParentLpuDims[0].partition, 
			lpuCount[1], lpuId[1], l, 0, 0);
	bLpuDims[1] = bParentLpuDims[1];
}

void mm::getcPartForSpaceALpu(PartDimension *cLpuDims, 
		PartDimension *cParentLpuDims, 
		int *lpuCount, int *lpuId, int k, int l) {
	cLpuDims[0].storage = cParentLpuDims[0].storage;
	cLpuDims[0].partition = block_size_getRange(cParentLpuDims[0].partition, 
			lpuCount[0], lpuId[0], k, 0, 0);
	cLpuDims[1].storage = cParentLpuDims[1].storage;
	cLpuDims[1].partition = block_size_getRange(cParentLpuDims[1].partition, 
			lpuCount[1], lpuId[1], l, 0, 0);
}

/*-----------------------------------------------------------------------------------
Functions for ArrayMetadata and EnvironmentLinks 
------------------------------------------------------------------------------------*/

mm::ArrayMetadata::ArrayMetadata() : Metadata() {
	setTaskName("Matrix Multiply");
}

/*-----------------------------------------------------------------------------------
function to initialize the content reference objects of LPSes
------------------------------------------------------------------------------------*/

void mm::initializeRootLPSContent(EnvironmentLinks *envLinks, ArrayMetadata *metadata) {
	spaceRootContent.a = envLinks->a;
	spaceRootContent.b = envLinks->b;
	spaceRootContent.c = allocate::allocateArray <float> (2, metadata->cDims);
	allocate::zeroFillArray <float> (0, spaceRootContent.c, 2, metadata->cDims);
}

void mm::initializeLPSesContents(ArrayMetadata *metadata) {
	//Processing Space A contents
	spaceAContent.a = spaceRootContent.a;
	spaceAContent.b = spaceRootContent.b;
	spaceAContent.c = spaceRootContent.c;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *mm::getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->threadNo = threadNo;
	threadIds->lpsCount = Space_Count;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	// for Space Root
	threadIds->ppuIds[Space_Root].lpsName = "Root";
	threadIds->ppuIds[Space_Root].groupId = 0;
	threadIds->ppuIds[Space_Root].groupSize = Total_Threads;
	threadIds->ppuIds[Space_Root].ppuCount = 1;
	threadIds->ppuIds[Space_Root].id = (threadNo == 0) ? 0 : INVALID_ID;

	int threadCount;
	int groupSize;
	int groupThreadId;

	// for Space A;
	threadIds->ppuIds[Space_A].lpsName = "A";
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
	threadLog << "set up parent LPS index map" << std::endl;
	threadLog.flush();
}

// Construction of task specific root LPU
void ThreadStateImpl::setRootLpu(Metadata *metadata) {

	ArrayMetadata *arrayMetadata = (ArrayMetadata*) metadata;

	SpaceRoot_LPU *lpu = new SpaceRoot_LPU;
	lpu->a = NULL;
	lpu->aPartDims[0].partition = arrayMetadata->aDims[0];
	lpu->aPartDims[0].storage = arrayMetadata->aDims[0].getNormalizedDimension();
	lpu->aPartDims[1].partition = arrayMetadata->aDims[1];
	lpu->aPartDims[1].storage = arrayMetadata->aDims[1].getNormalizedDimension();

	lpu->b = NULL;
	lpu->bPartDims[0].partition = arrayMetadata->bDims[0];
	lpu->bPartDims[0].storage = arrayMetadata->bDims[0].getNormalizedDimension();
	lpu->bPartDims[1].partition = arrayMetadata->bDims[1];
	lpu->bPartDims[1].storage = arrayMetadata->bDims[1].getNormalizedDimension();

	lpu->c = NULL;
	lpu->cPartDims[0].partition = arrayMetadata->cDims[0];
	lpu->cPartDims[0].storage = arrayMetadata->cDims[0].getNormalizedDimension();
	lpu->cPartDims[1].partition = arrayMetadata->cDims[1];
	lpu->cPartDims[1].storage = arrayMetadata->cDims[1].getNormalizedDimension();

	lpu->setValidBit(true);
	lpsStates[Space_Root]->lpu = lpu;
	threadLog << "set up root LPU" << std::endl;
	threadLog.flush();
}

// Initialization of LPU pointers of different LPSes
void ThreadStateImpl::initializeLPUs() {
	lpsStates[Space_A]->lpu = new SpaceA_LPU;
	lpsStates[Space_A]->lpu->setValidBit(false);
	threadLog << "initialized LPU pointers" << std::endl;
	threadLog.flush();
}

// Implementation of task specific compute-LPU-Count function 
int *ThreadStateImpl::computeLpuCounts(int lpsId) {
	if (lpsId == Space_Root) {
		return NULL;
	}
	if (lpsId == Space_A) {
		int ppuCount = threadIds->ppuIds[Space_A].ppuCount;
		SpaceRoot_LPU *spaceRootLpu
				 = (SpaceRoot_LPU*) lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceA(ppuCount, 
				spaceRootLpu->cPartDims[0].partition, 
				partitionArgs[0], 
				spaceRootLpu->cPartDims[1].partition, 
				partitionArgs[1]);
	}
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_A) {
		SpaceRoot_LPU *spaceRootLpu
				 = (SpaceRoot_LPU*) lpsStates[Space_Root]->lpu;
		SpaceA_LPU *currentLpu
				 = (SpaceA_LPU*) lpsStates[Space_A]->lpu;
		currentLpu->lpuId[0] = nextLpuId[0];
		currentLpu->lpuId[1] = nextLpuId[1];
		currentLpu->a = spaceAContent.a;
		getaPartForSpaceALpu(currentLpu->aPartDims, 
				spaceRootLpu->aPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		currentLpu->b = spaceAContent.b;
		getbPartForSpaceALpu(currentLpu->bPartDims, 
				spaceRootLpu->bPartDims, lpuCounts, nextLpuId, 
				partitionArgs[1]);
		currentLpu->c = spaceAContent.c;
		getcPartForSpaceALpu(currentLpu->cPartDims, 
				spaceRootLpu->cPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0], partitionArgs[1]);
		currentLpu->setValidBit(true);
		return currentLpu;
	}
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void mm::initializeTask(ArrayMetadata *arrayMetadata, 
		EnvironmentLinks environmentLinks, 
		TaskGlobals *taskGlobals, 
		ThreadLocals *threadLocals, 
		MMPartition partition) {

	arrayMetadata->aDims[0] = environmentLinks.aDims[0];
	arrayMetadata->aDims[1] = environmentLinks.aDims[1];
	arrayMetadata->bDims[0] = environmentLinks.bDims[0];
	arrayMetadata->bDims[1] = environmentLinks.bDims[1];
	arrayMetadata->cDims[0] = arrayMetadata->aDims[0];
	arrayMetadata->cDims[1] = arrayMetadata->bDims[1];
}

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void mm::mm_function0(SpaceA_LPU *lpu, 
		ArrayMetadata *arrayMetadata, 
		TaskGlobals *taskGlobals, 
		ThreadLocals *threadLocals, MMPartition partition) {

	//-------------------- Local Copies of Metadata -----------------------------

	Dimension aPartDims[2];
	aPartDims[0] = lpu->aPartDims[0].partition;
	aPartDims[1] = lpu->aPartDims[1].partition;
	Dimension aStoreDims[2];
	aStoreDims[0] = lpu->aPartDims[0].storage;
	aStoreDims[1] = lpu->aPartDims[1].storage;
	Dimension bPartDims[2];
	bPartDims[0] = lpu->bPartDims[0].partition;
	bPartDims[1] = lpu->bPartDims[1].partition;
	Dimension bStoreDims[2];
	bStoreDims[0] = lpu->bPartDims[0].storage;
	bStoreDims[1] = lpu->bPartDims[1].storage;
	Dimension cPartDims[2];
	cPartDims[0] = lpu->cPartDims[0].partition;
	cPartDims[1] = lpu->cPartDims[1].partition;
	Dimension cStoreDims[2];
	cStoreDims[0] = lpu->cPartDims[0].storage;
	cStoreDims[1] = lpu->cPartDims[1].storage;

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
		int i_c_0 = (i - cPartDims[0].getPositiveRange().min) * cStoreDims[1].getLength();
		int i_a_0 = (i - aPartDims[0].getPositiveRange().min) * aStoreDims[1].getLength();
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
			int j_c_1 = (j - cPartDims[1].getPositiveRange().min);
			int j_b_1 = (j - bPartDims[1].getPositiveRange().min);
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
				int k_a_1 = (k - aPartDims[1].getPositiveRange().min);
				int k_b_0 = (k - bPartDims[0].getPositiveRange().min) * bStoreDims[1].getLength();
				lpu->c[i_c_0 + j_c_1] = lpu->c[i_c_0 + j_c_1] + lpu->a[i_a_0 + k_a_1] * lpu->b[k_b_0 + j_b_1];
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

void mm::run(ArrayMetadata *arrayMetadata, 
		TaskGlobals *taskGlobals, 
		ThreadLocals *threadLocals, 
		MMPartition partition, ThreadStateImpl *threadState) {

	// set the root LPU in the thread state so that calculation can start
	threadState->setRootLpu(arrayMetadata);

	{ // scope entrance for iterating LPUs of Space A
	int spaceALpuId = INVALID_ID;
	int spaceAIteration = 0;
	SpaceA_LPU *spaceALpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState->getNextLpu(Space_A, Space_Root, spaceALpuId)) != NULL) {
		spaceALpu = (SpaceA_LPU*) lpu;
		if (threadState->isValidPpu(Space_A)) {
			// invoking user computation
			mm_function0(spaceALpu, 
					arrayMetadata,
					taskGlobals,
					threadLocals, partition);
			threadState->logExecution("mm_function0", Space_A);
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
	ArrayMetadata *metadata = new ArrayMetadata;
	MMEnvironment environment;
	MMPartition partition;

	// creating a program log file
	std::cout << "Creating diagnostic log: it-program.log\n";
	std::ofstream logFile;
	logFile.open("it-program.log");

	// initializing variables that are environmental links 
	std::cout << "initializing environmental links\n";
	inprompt::readArrayDimensionInfo("a", 2, envLinks.aDims);
	envLinks.a = allocate::allocateArray <float> (2, envLinks.aDims);
	allocate::randomFillPrimitiveArray <float> (envLinks.a, 2, envLinks.aDims);
	inprompt::readArrayDimensionInfo("b", 2, envLinks.bDims);
	envLinks.b = allocate::allocateArray <float> (2, envLinks.bDims);
	allocate::randomFillPrimitiveArray <float> (envLinks.b, 2, envLinks.bDims);

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
	initializeTask(metadata, envLinks, &taskGlobals, &threadLocals, partition);

	// setting the global metadata variable
	arrayMetadata = *metadata;

	// allocating memories for data structures
	std::cout << "Allocating memories\n";
	mm::initializeRootLPSContent(&envLinks, metadata);
	mm::initializeLPSesContents(metadata);

	// declaring and initializing state variables for threads 
	ThreadLocals *threadLocalsList[Total_Threads];
	for (int i = 0; i < Total_Threads; i++) {
		threadLocalsList[i] = new ThreadLocals;
		*threadLocalsList[i] = threadLocals;
	}
	int lpsDimensions[Space_Count];
	lpsDimensions[Space_Root] = 0;
	lpsDimensions[Space_A] = 2;
	std::cout << "generating PPU Ids for threads\n";
	ThreadIds *threadIdsList[Total_Threads];
	for (int i = 0; i < Total_Threads; i++) {
		threadIdsList[i] = getPpuIdsForThread(i);
		threadIdsList[i]->print(logFile);
	}
	std::cout << "initiating thread-states\n";
	ThreadStateImpl *threadStateList[Total_Threads];
	for (int i = 0; i < Total_Threads; i++) {
		threadStateList[i] = new ThreadStateImpl(Space_Count, 
				lpsDimensions, partitionArgs, threadIdsList[i]);
		threadStateList[i]->initiateLogFile("mm");
		threadStateList[i]->initializeLPUs();
		threadStateList[i]->setLpsParentIndexMap();
	}

	// starting threads
	std::cout << "starting threads\n";
	for (int i = 0; i < Total_Threads; i++) {
		run(metadata, &taskGlobals, threadLocalsList[i], partition, threadStateList[i]);
	}

	logFile.close();
	return 0;
}
