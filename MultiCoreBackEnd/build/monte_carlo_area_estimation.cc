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

PartitionDimension **mcae::getsubarea_estimatePartForSpaceBLpu(PartitionDimension **subarea_estimateParentLpuDims, 
		int *lpuCount, int *lpuId) {
	PartitionDimension **subarea_estimateLpuDims = new PartitionDimension*[2];
	subarea_estimateLpuDims[0] = new PartitionDimension;
	subarea_estimateLpuDims[0]->storageDim = subarea_estimateParentLpuDims[0]->partitionDim;
	subarea_estimateLpuDims[0]->partitionDim = block_size_getRange(*subarea_estimateParentLpuDims[0]->partitionDim, 
			lpuCount[0], lpuId[0], 1, 0, 0);
	subarea_estimateLpuDims[1] = new PartitionDimension;
	subarea_estimateLpuDims[1]->storageDim = subarea_estimateParentLpuDims[1]->partitionDim;
	subarea_estimateLpuDims[1]->partitionDim = block_size_getRange(*subarea_estimateParentLpuDims[1]->partitionDim, 
			lpuCount[1], lpuId[1], 1, 0, 0);
	return subarea_estimateLpuDims;
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

	lpu->subarea_estimate = NULL;
	lpu->subarea_estimatePartDims = new PartitionDimension*[2];
	lpu->subarea_estimatePartDims[0] = new PartitionDimension;
	lpu->subarea_estimatePartDims[0]->storageDim = lpu->subarea_estimatePartDims[0]->partitionDim
			= &arrayMetadata.subarea_estimateDims[0];
	lpu->subarea_estimatePartDims[1] = new PartitionDimension;
	lpu->subarea_estimatePartDims[1]->storageDim = lpu->subarea_estimatePartDims[1]->partitionDim
			= &arrayMetadata.subarea_estimateDims[1];

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
		currentLpu->subarea_estimate = NULL;
		currentLpu->subarea_estimatePartDims = spaceRootLpu->subarea_estimatePartDims;
		return currentLpu;
	}
	if (lpsId == Space_B) {
		SpaceRoot_LPU *spaceRootLpu = (SpaceRoot_LPU*) 
				lpsStates[Space_Root]->lpu;
		SpaceC_LPU *spaceCLpu = (SpaceC_LPU*) 
				lpsStates[Space_C]->lpu;
		SpaceB_LPU *currentLpu = new SpaceB_LPU;
		currentLpu->grid = NULL;
		currentLpu->gridPartDims = getgridPartForSpaceBLpu(
				spaceRootLpu->gridPartDims, lpuCounts, nextLpuId);
		currentLpu->point_placements = NULL;
		currentLpu->point_placementsPartDims = getpoint_placementsPartForSpaceBLpu(
				spaceRootLpu->point_placementsPartDims, lpuCounts, nextLpuId);
		currentLpu->subarea_estimate = NULL;
		currentLpu->subarea_estimatePartDims = getsubarea_estimatePartForSpaceBLpu(
				spaceCLpu->subarea_estimatePartDims, lpuCounts, nextLpuId);
		return currentLpu;
	}
	if (lpsId == Space_A) {
		SpaceB_LPU *spaceBLpu = (SpaceB_LPU*) 
				lpsStates[Space_B]->lpu;
		SpaceA_LPU *currentLpu = new SpaceA_LPU;
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
		float cell_size) {

	arrayMetadata.gridDims[0] = environmentLinks.gridDims[0];
	arrayMetadata.gridDims[1] = environmentLinks.gridDims[1];
	taskGlobals.shape = environmentLinks.shape;
	taskGlobals.precision = precision;
	taskGlobals.cell_size = cell_size;
	arrayMetadata.subarea_estimateDims[0] = arrayMetadata.gridDims[0];
	arrayMetadata.subarea_estimateDims[1] = arrayMetadata.gridDims[1];
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
	Dimension point_placementsPartDims[3];
	point_placementsPartDims[0] = *lpu.point_placementsPartDims[0]->storageDim;
	point_placementsPartDims[1] = *lpu.point_placementsPartDims[1]->storageDim;
	point_placementsPartDims[2] = *lpu.point_placementsPartDims[2]->storageDim;
	Dimension subarea_estimatePartDims[2];
	subarea_estimatePartDims[0] = *lpu.subarea_estimatePartDims[0]->storageDim;
	subarea_estimatePartDims[1] = *lpu.subarea_estimatePartDims[1]->storageDim;

	//declare the local variables of this compute stage
	int external_points;
	int internal_points;
	int local_points;
	float total_inside;
	int total_outside;
	float total_points;
}

void mcae::Estimate_Total_Area(SpaceC_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition) {

	//create local variables for array dimensions 
	Dimension subarea_estimatePartDims[2];
	subarea_estimatePartDims[0] = *lpu.subarea_estimatePartDims[0]->storageDim;
	subarea_estimatePartDims[1] = *lpu.subarea_estimatePartDims[1]->storageDim;
}

