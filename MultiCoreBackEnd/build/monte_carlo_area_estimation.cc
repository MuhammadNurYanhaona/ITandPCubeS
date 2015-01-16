/*-----------------------------------------------------------------------------------
header file for the task
------------------------------------------------------------------------------------*/
#include "monte_carlo_area_estimation.h"

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

using namespace mcae;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/

int *mcae::getLPUsCountOfSpaceB(int ppuCount, Dimension point_placementsDim1, Dimension point_placementsDim2) {
	int *count = new int[2];
	count[0] = block_size_partitionCount(point_placementsDim1, ppuCount, 1);
	count[1] = block_size_partitionCount(point_placementsDim2, ppuCount, 1);
	return count;
}

int *mcae::getLPUsCountOfSpaceA(int ppuCount, Dimension point_placementsDim3, int p) {
	int *count = new int[1];
	count[0] = block_count_partitionCount(point_placementsDim3, ppuCount, p);
	return count;
}

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/

PartitionDimension **mcae::getgridPartForSpaceBLpu(PartitionDimension **gridParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **gridLpuDims = new PartitionDimension*[2];
	gridLpuDims[0] = new PartitionDimension;
	gridLpuDims[0]->storageDim = gridParentLpuDims[0]->partitionDim;
	gridLpuDims[0]->partitionDim = block_size_getRange(*gridParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	gridLpuDims[1] = new PartitionDimension;
	gridLpuDims[1]->storageDim = gridParentLpuDims[1]->partitionDim;
	gridLpuDims[1]->partitionDim = block_size_getRange(*gridParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	return gridLpuDims;
}

PartitionDimension **mcae::getlocal_prePartForSpaceBLpu(PartitionDimension **local_preParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **local_preLpuDims = new PartitionDimension*[2];
	local_preLpuDims[0] = new PartitionDimension;
	local_preLpuDims[0]->storageDim = local_preParentLpuDims[0]->partitionDim;
	local_preLpuDims[0]->partitionDim = block_size_getRange(*local_preParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	local_preLpuDims[1] = new PartitionDimension;
	local_preLpuDims[1]->storageDim = local_preParentLpuDims[1]->partitionDim;
	local_preLpuDims[1]->partitionDim = block_size_getRange(*local_preParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	return local_preLpuDims;
}

PartitionDimension **mcae::getpoint_placementsPartForSpaceBLpu(PartitionDimension **point_placementsParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **point_placementsLpuDims = new PartitionDimension*[3];
	point_placementsLpuDims[0] = new PartitionDimension;
	point_placementsLpuDims[0]->storageDim = point_placementsParentLpuDims[0]->partitionDim;
	point_placementsLpuDims[0]->partitionDim = block_size_getRange(*point_placementsParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	point_placementsLpuDims[1] = new PartitionDimension;
	point_placementsLpuDims[1]->storageDim = point_placementsParentLpuDims[1]->partitionDim;
	point_placementsLpuDims[1]->partitionDim = block_size_getRange(*point_placementsParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	point_placementsLpuDims[2] = point_placementsParentLpuDims[2];
	return point_placementsLpuDims;
}

PartitionDimension **mcae::getstatsPartForSpaceBLpu(PartitionDimension **statsParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **statsLpuDims = new PartitionDimension*[2];
	statsLpuDims[0] = new PartitionDimension;
	statsLpuDims[0]->storageDim = statsParentLpuDims[0]->partitionDim;
	statsLpuDims[0]->partitionDim = block_size_getRange(*statsParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	statsLpuDims[1] = new PartitionDimension;
	statsLpuDims[1]->storageDim = statsParentLpuDims[1]->partitionDim;
	statsLpuDims[1]->partitionDim = block_size_getRange(*statsParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	return statsLpuDims;
}

PartitionDimension **mcae::getsub_estimatesPartForSpaceBLpu(PartitionDimension **sub_estimatesParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **sub_estimatesLpuDims = new PartitionDimension*[2];
	sub_estimatesLpuDims[0] = new PartitionDimension;
	sub_estimatesLpuDims[0]->storageDim = sub_estimatesParentLpuDims[0]->partitionDim;
	sub_estimatesLpuDims[0]->partitionDim = block_size_getRange(*sub_estimatesParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	sub_estimatesLpuDims[1] = new PartitionDimension;
	sub_estimatesLpuDims[1]->storageDim = sub_estimatesParentLpuDims[1]->partitionDim;
	sub_estimatesLpuDims[1]->partitionDim = block_size_getRange(*sub_estimatesParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	return sub_estimatesLpuDims;
}

PartitionDimension **mcae::getpoint_placementsPartForSpaceALpu(PartitionDimension **point_placementsParentLpuDims, 
		int *lpuCount, int *lpuId, int p) {
	PartitionDimension **point_placementsLpuDims = new PartitionDimension*[3];
	point_placementsLpuDims[0] = point_placementsParentLpuDims[0];
	point_placementsLpuDims[1] = point_placementsParentLpuDims[1];
	point_placementsLpuDims[2] = new PartitionDimension;
	point_placementsLpuDims[2]->storageDim = point_placementsParentLpuDims[2]->partitionDim;
	point_placementsLpuDims[2]->partitionDim = block_count_getRange(*point_placementsParentLpuDims[2]->partitionDim, 
			lpuCount[0], lpuId[0], p, 0, 0);
	return point_placementsLpuDims;
}

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/

ThreadIds *mcae::getPpuIdsForThread(int threadNo)  {

	ThreadIds *threadIds = new ThreadIds;
	threadIds->ppuIds = new PPU_Ids[Space_Count];
	int idsArray[Space_Count];
	idsArray[Space_Root] = threadNo;

	int threadCount;
	int groupSize;
	int groupThreadId;

	// for Space C;
	threadCount = Total_Threads;
	groupSize = threadCount;
	groupThreadId = idsArray[Space_Root] % groupSize;
	threadIds->ppuIds[Space_C].groupId = idsArray[Space_Root] / groupSize;
	threadIds->ppuIds[Space_C].ppuCount = 1;
	threadIds->ppuIds[Space_C].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_C].id
			= threadIds->ppuIds[Space_C].groupId;
	else threadIds->ppuIds[Space_C].id = INVALID_ID;
	idsArray[Space_C] = groupThreadId;

	// for Space B;
	threadCount = threadIds->ppuIds[Space_C].groupSize;
	groupSize = threadCount / 16;
	groupThreadId = idsArray[Space_C] % groupSize;
	threadIds->ppuIds[Space_B].groupId = idsArray[Space_C] / groupSize;
	threadIds->ppuIds[Space_B].ppuCount = 16;
	threadIds->ppuIds[Space_B].groupSize = groupSize;
	if (groupThreadId == 0) threadIds->ppuIds[Space_B].id
			= threadIds->ppuIds[Space_B].groupId;
	else threadIds->ppuIds[Space_B].id = INVALID_ID;
	idsArray[Space_B] = groupThreadId;

	// for Space A;
	threadCount = threadIds->ppuIds[Space_B].groupSize;
	groupSize = threadCount / 4;
	groupThreadId = idsArray[Space_B] % groupSize;
	threadIds->ppuIds[Space_A].groupId = idsArray[Space_B] / groupSize;
	threadIds->ppuIds[Space_A].ppuCount = 4;
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
	lpsParentIndexMap[Space_C] = Space_Root;
	lpsParentIndexMap[Space_B] = Space_C;
	lpsParentIndexMap[Space_A] = Space_B;
}

// Construction of task specific root LPU
void ThreadStateImpl::setRootLpu() {
	SpaceRoot_LPU *lpu = new SpaceRoot_LPU;
	lpu->grid = NULL;
	lpu->gridPartDims = new PartitionDimension*[2];
	lpu->gridPartDims[0] = new PartitionDimension;
	lpu->gridPartDims[0]->storageDim = lpu->gridPartDims[0]->partitionDim
			= &arrayMetadata.gridDims[0];
	lpu->gridPartDims[1] = new PartitionDimension;
	lpu->gridPartDims[1]->storageDim = lpu->gridPartDims[1]->partitionDim
			= &arrayMetadata.gridDims[1];

	lpu->local_pre = NULL;
	lpu->local_prePartDims = new PartitionDimension*[2];
	lpu->local_prePartDims[0] = new PartitionDimension;
	lpu->local_prePartDims[0]->storageDim = lpu->local_prePartDims[0]->partitionDim
			= &arrayMetadata.local_preDims[0];
	lpu->local_prePartDims[1] = new PartitionDimension;
	lpu->local_prePartDims[1]->storageDim = lpu->local_prePartDims[1]->partitionDim
			= &arrayMetadata.local_preDims[1];

	lpu->point_placements = NULL;
	lpu->point_placementsPartDims = new PartitionDimension*[3];
	lpu->point_placementsPartDims[0] = new PartitionDimension;
	lpu->point_placementsPartDims[0]->storageDim = lpu->point_placementsPartDims[0]->partitionDim
			= &arrayMetadata.point_placementsDims[0];
	lpu->point_placementsPartDims[1] = new PartitionDimension;
	lpu->point_placementsPartDims[1]->storageDim = lpu->point_placementsPartDims[1]->partitionDim
			= &arrayMetadata.point_placementsDims[1];
	lpu->point_placementsPartDims[2] = new PartitionDimension;
	lpu->point_placementsPartDims[2]->storageDim = lpu->point_placementsPartDims[2]->partitionDim
			= &arrayMetadata.point_placementsDims[2];

	lpu->stats = NULL;
	lpu->statsPartDims = new PartitionDimension*[2];
	lpu->statsPartDims[0] = new PartitionDimension;
	lpu->statsPartDims[0]->storageDim = lpu->statsPartDims[0]->partitionDim
			= &arrayMetadata.statsDims[0];
	lpu->statsPartDims[1] = new PartitionDimension;
	lpu->statsPartDims[1]->storageDim = lpu->statsPartDims[1]->partitionDim
			= &arrayMetadata.statsDims[1];

	lpu->sub_estimates = NULL;
	lpu->sub_estimatesPartDims = new PartitionDimension*[2];
	lpu->sub_estimatesPartDims[0] = new PartitionDimension;
	lpu->sub_estimatesPartDims[0]->storageDim = lpu->sub_estimatesPartDims[0]->partitionDim
			= &arrayMetadata.sub_estimatesDims[0];
	lpu->sub_estimatesPartDims[1] = new PartitionDimension;
	lpu->sub_estimatesPartDims[1]->storageDim = lpu->sub_estimatesPartDims[1]->partitionDim
			= &arrayMetadata.sub_estimatesDims[1];

	lpsStates[Space_Root]->lpu = lpu;
}

// Implementation of task specific compute-LPU-Count function 
int *ThreadStateImpl::computeLpuCounts(int lpsId) {
	if (lpsId == Space_Root) {
		return NULL;
	}
	if (lpsId == Space_C) {
		return NULL;
	}
	if (lpsId == Space_B) {
		int ppuCount = threadIds->ppuIds[Space_B].ppuCount;
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		return getLPUsCountOfSpaceB(ppuCount, 
				*spaceRootLpu->point_placementsPartDims[0]->partitionDim, 
				*spaceRootLpu->point_placementsPartDims[1]->partitionDim);
	}
	if (lpsId == Space_A) {
		int ppuCount = threadIds->ppuIds[Space_A].ppuCount;
		SpaceB_LPU *spaceBLpu = (SpaceB_LPU*) 
				lpsStates[Space_B]->lpu;
		return getLPUsCountOfSpaceA(ppuCount, 
				*spaceBLpu->point_placementsPartDims[2]->partitionDim, 
				partitionArgs[0]);
	}
	return NULL;
}

// Implementation of task specific compute-Next-LPU function 
LPU *ThreadStateImpl::computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId) {
	if (lpsId == Space_C) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceC_LPU *currentLpu = new SpaceC_LPU;
		currentLpu->sub_estimates = NULL;
		currentLpu->sub_estimatesPartDims = spaceRootLpu->sub_estimatesPartDims;
		return currentLpu;
	}
	if (lpsId == Space_B) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceC_LPU *spaceCLpu = (SpaceC_LPU*) 
				lpsStates[Space_C]->lpu;
		SpaceB_LPU *currentLpu = new SpaceB_LPU;
		currentLpu->lpuId[0] = nextLpuId[0];
		currentLpu->lpuId[1] = nextLpuId[1];
		currentLpu->grid = NULL;
		currentLpu->gridPartDims = getgridPartForSpaceBLpu(
				spaceRootLpu->gridPartDims, lpuCounts, nextLpuId);
		currentLpu->local_pre = NULL;
		currentLpu->local_prePartDims = getlocal_prePartForSpaceBLpu(
				spaceRootLpu->local_prePartDims, lpuCounts, nextLpuId);
		currentLpu->point_placements = NULL;
		currentLpu->point_placementsPartDims = getpoint_placementsPartForSpaceBLpu(
				spaceRootLpu->point_placementsPartDims, lpuCounts, nextLpuId);
		currentLpu->stats = NULL;
		currentLpu->statsPartDims = getstatsPartForSpaceBLpu(
				spaceRootLpu->statsPartDims, lpuCounts, nextLpuId);
		currentLpu->sub_estimates = NULL;
		currentLpu->sub_estimatesPartDims = getsub_estimatesPartForSpaceBLpu(
				spaceCLpu->sub_estimatesPartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	if (lpsId == Space_A) {
		SpaceB_LPU *spaceBLpu = (SpaceB_LPU*) 
				lpsStates[Space_B]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
		currentLpu->lpuId[0] = nextLpuId[0];
		currentLpu->grid = NULL;
		currentLpu->gridPartDims = spaceBLpu->gridPartDims;
		currentLpu->point_placements = NULL;
		currentLpu->point_placementsPartDims = getpoint_placementsPartForSpaceALpu(
				spaceBLpu->point_placementsPartDims, lpuCounts, nextLpuId, 
				partitionArgs[0]);
		return currentLpu;
	}
	return NULL;
}

/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/

void mcae::initializeTask(TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MCAEPartition partition, 
		float precision, 
		float cell_size, 
		int maxIterations) {

	arrayMetadata.gridDims[0] = environmentLinks.gridDims[0];
	arrayMetadata.gridDims[1] = environmentLinks.gridDims[1];
	taskGlobals.shape = environmentLinks.shape;
	taskGlobals.precision = precision;
	taskGlobals.cell_size = cell_size;
	taskGlobals.maxIterations = maxIterations;
	arrayMetadata.local_preDims[0] = arrayMetadata.gridDims[0];
	arrayMetadata.local_preDims[1] = arrayMetadata.gridDims[1];
	arrayMetadata.statsDims[0] = arrayMetadata.local_preDims[0];
	arrayMetadata.statsDims[1] = arrayMetadata.local_preDims[1];
	arrayMetadata.sub_estimatesDims[0] = arrayMetadata.statsDims[0];
	arrayMetadata.sub_estimatesDims[1] = arrayMetadata.statsDims[1];
	arrayMetadata.point_placementsDims[0] = arrayMetadata.gridDims[0];
	arrayMetadata.point_placementsDims[1] = arrayMetadata.gridDims[1];
	arrayMetadata.point_placementsDims[2].range.min = 0;
	arrayMetadata.point_placementsDims[2].range.max = partition.p - 1;
}

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void mcae::Calculate_Point_Position(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition) {

	//create local variables for array dimensions 
	Dimension gridPartDims[2];
	gridPartDims[0] = *lpu.gridPartDims[0]->storageDim;
	gridPartDims[1] = *lpu.gridPartDims[1]->storageDim;
	Dimension point_placementsPartDims[3];
	point_placementsPartDims[0] = *lpu.point_placementsPartDims[0]->storageDim;
	point_placementsPartDims[1] = *lpu.point_placementsPartDims[1]->storageDim;
	point_placementsPartDims[2] = *lpu.point_placementsPartDims[2]->storageDim;

	//declare the local variables of this compute stage
	Point point;
}

void mcae::Refine_Subarea_Estimate(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition) {

	//create local variables for array dimensions 
	Dimension gridPartDims[2];
	gridPartDims[0] = *lpu.gridPartDims[0]->storageDim;
	gridPartDims[1] = *lpu.gridPartDims[1]->storageDim;
	Dimension local_prePartDims[2];
	local_prePartDims[0] = *lpu.local_prePartDims[0]->storageDim;
	local_prePartDims[1] = *lpu.local_prePartDims[1]->storageDim;
	Dimension point_placementsPartDims[3];
	point_placementsPartDims[0] = *lpu.point_placementsPartDims[0]->storageDim;
	point_placementsPartDims[1] = *lpu.point_placementsPartDims[1]->storageDim;
	point_placementsPartDims[2] = *lpu.point_placementsPartDims[2]->storageDim;
	Dimension statsPartDims[2];
	statsPartDims[0] = *lpu.statsPartDims[0]->storageDim;
	statsPartDims[1] = *lpu.statsPartDims[1]->storageDim;
	Dimension sub_estimatesPartDims[2];
	sub_estimatesPartDims[0] = *lpu.sub_estimatesPartDims[0]->storageDim;
	sub_estimatesPartDims[1] = *lpu.sub_estimatesPartDims[1]->storageDim;

	//declare the local variables of this compute stage
	int external_points;
	int internal_points;
	int local_points;
	float oldEstimate;
	float total_points;
}

void mcae::Estimate_Total_Area(SpaceC_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition) {

	//create local variables for array dimensions 
	Dimension sub_estimatesPartDims[2];
	sub_estimatesPartDims[0] = *lpu.sub_estimatesPartDims[0]->storageDim;
	sub_estimatesPartDims[1] = *lpu.sub_estimatesPartDims[1]->storageDim;
}

/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void mcae::run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MCAEPartition partition, ThreadStateImpl threadState) {

	// set the root LPU in the thread state so that calculation can start
	threadState.setRootLpu();

	{ // scope entrance for iterating LPUs of Space B
	int spaceBLpuId = INVALID_ID;
	int spaceBIteration = 0;
	SpaceB_LPU *spaceBLpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_B, Space_Root, spaceBLpuId)) != NULL) {
		spaceBLpu = (SpaceB_LPU*) lpu;
		while (threadLocals.t.current < taskGlobals.maxIterations) {
			{ // scope entrance for iterating LPUs of Space A
			int spaceALpuId = INVALID_ID;
			int spaceAIteration = 0;
			SpaceA_LPU *spaceALpu = NULL;
			LPU *lpu = NULL;
			while((lpu = threadState.getNextLpu(Space_A, Space_B, spaceALpuId)) != NULL) {
				spaceALpu = (SpaceA_LPU*) lpu;
				if (threadState.isValidPpu(Space_A)) {
					// invoking user computation
					Calculate_Point_Position(*spaceALpu, 
							arrayMetadata,
							taskGlobals,
							threadLocals, partition);
				}
				spaceALpuId = spaceALpu->id;
				spaceAIteration++;
			}
			threadState.removeIterationBound(Space_B);
			} // scope exit for iterating LPUs of Space A
			// invoking user computation
			Refine_Subarea_Estimate(*spaceBLpu, 
					arrayMetadata,
					taskGlobals,
					threadLocals, partition);
		}
		spaceBLpuId = spaceBLpu->id;
		spaceBIteration++;
	}
	} // scope exit for iterating LPUs of Space B
	{ // scope entrance for iterating LPUs of Space C
	int spaceCLpuId = INVALID_ID;
	int spaceCIteration = 0;
	SpaceC_LPU *spaceCLpu = NULL;
	LPU *lpu = NULL;
	while((lpu = threadState.getNextLpu(Space_C, Space_Root, spaceCLpuId)) != NULL) {
		spaceCLpu = (SpaceC_LPU*) lpu;
		// invoking user computation
		Estimate_Total_Area(*spaceCLpu, 
				arrayMetadata,
				taskGlobals,
				threadLocals, partition);
		spaceCLpuId = spaceCLpu->id;
		spaceCIteration++;
	}
	} // scope exit for iterating LPUs of Space C
}

