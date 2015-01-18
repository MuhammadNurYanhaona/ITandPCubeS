#ifndef _H_mcae
#define _H_mcae

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

namespace mcae {

/*-----------------------------------------------------------------------------------
constants for LPSes
------------------------------------------------------------------------------------*/
const int Space_Root = 0;
const int Space_C = 1;
const int Space_B = 2;
const int Space_A = 3;
const int Space_Count = 4;

/*-----------------------------------------------------------------------------------
constants for PPS counts
------------------------------------------------------------------------------------*/
const int Space_5_PPUs = 1;
const int Space_4_Par_5_PPUs = 2;
const int Space_3_Par_4_PPUs = 4;
const int Space_2_Par_3_PPUs = 2;
const int Space_1_Par_2_PPUs = 4;

/*-----------------------------------------------------------------------------------
constants for total and par core thread counts
------------------------------------------------------------------------------------*/
const int Total_Threads = 64;
const int Threads_Par_Core = 4;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/
int *getLPUsCountOfSpaceB(int ppuCount, Dimension point_placementsDim1, Dimension point_placementsDim2);
int *getLPUsCountOfSpaceA(int ppuCount, Dimension point_placementsDim3, int p);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getgridPartForSpaceBLpu(PartitionDimension **gridParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getlocal_prePartForSpaceBLpu(PartitionDimension **local_preParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getpoint_placementsPartForSpaceBLpu(PartitionDimension **point_placementsParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getstatsPartForSpaceBLpu(PartitionDimension **statsParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getsub_estimatesPartForSpaceBLpu(PartitionDimension **sub_estimatesParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getpoint_placementsPartForSpaceALpu(PartitionDimension **point_placementsParentLpuDims, 
		int *lpuCount, int *lpuId, int p);

/*-----------------------------------------------------------------------------------
Data structures representing LPS and LPU contents 
------------------------------------------------------------------------------------*/

class SpaceRoot_Content {
  public:
	Rectangle *grid;
	float *local_pre;
	int *point_placements;
	PlacementStatistic *stats;
	float *sub_estimates;
};

class SpaceRoot_LPU : public LPU {
  public:
	Rectangle *grid;
	PartitionDimension **gridPartDims;
	float *local_pre;
	PartitionDimension **local_prePartDims;
	int *point_placements;
	PartitionDimension **point_placementsPartDims;
	PlacementStatistic *stats;
	PartitionDimension **statsPartDims;
	float *sub_estimates;
	PartitionDimension **sub_estimatesPartDims;
};

class SpaceC_Content {
  public:
	float *sub_estimates;
};

class SpaceC_LPU : public LPU {
  public:
	float *sub_estimates;
	PartitionDimension **sub_estimatesPartDims;
};

class SpaceB_Content {
  public:
	Rectangle *grid;
	float *local_pre;
	int *point_placements;
	PlacementStatistic *stats;
	float *sub_estimates;
};

class SpaceB_LPU : public LPU {
  public:
	Rectangle *grid;
	PartitionDimension **gridPartDims;
	float *local_pre;
	PartitionDimension **local_prePartDims;
	int *point_placements;
	PartitionDimension **point_placementsPartDims;
	PlacementStatistic *stats;
	PartitionDimension **statsPartDims;
	float *sub_estimates;
	PartitionDimension **sub_estimatesPartDims;
	int lpuId[2];
};

class SpaceA_Content {
  public:
	Rectangle *grid;
	int *point_placements;
};

class SpaceA_LPU : public LPU {
  public:
	Rectangle *grid;
	PartitionDimension **gridPartDims;
	int *point_placements;
	PartitionDimension **point_placementsPartDims;
	int lpuId[1];
};

/*-----------------------------------------------------------------------------------
Data structures for Array-Metadata and Environment-Links 
------------------------------------------------------------------------------------*/

class ArrayMetadata {
  public:
	Dimension gridDims[2];
	Dimension local_preDims[2];
	Dimension point_placementsDims[3];
	Dimension statsDims[2];
	Dimension sub_estimatesDims[2];
};
ArrayMetadata arrayMetadata;

class EnvironmentLinks {
  public:
	Rectangle *grid;
	Dimension gridDims[2];
	std::vector<Coefficients> shape;
};
EnvironmentLinks environmentLinks;

/*-----------------------------------------------------------------------------------
Data structures for Task-Global and Thread-Local scalar variables
------------------------------------------------------------------------------------*/

class TaskGlobals {
  public:
	float area;
	float cell_size;
	float precision;
	std::vector<Coefficients> shape;
};

class ThreadLocals {
  public:
};

/*-----------------------------------------------------------------------------------
function to generate PPU IDs and PPU group IDs for a thread
------------------------------------------------------------------------------------*/
ThreadIds *getPpuIdsForThread(int threadNo);

/*-----------------------------------------------------------------------------------
Thread-State implementation class for the task
------------------------------------------------------------------------------------*/

class ThreadStateImpl : public ThreadState {
  public:
	void setLpsParentIndexMap();
        void setRootLpu();
        int *computeLpuCounts(int lpsId);
        LPU *computeNextLpu(int lpsId, int *lpuCounts, int *nextLpuId);
};


/*-----------------------------------------------------------------------------------
function for the initialize block
------------------------------------------------------------------------------------*/
void initializeTask(TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MCAEPartition partition, 
		float precision, 
		float cell_size);

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void Calculate_Point_Position(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition);

void Refine_Subarea_Estimate(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition);

void Estimate_Total_Area(SpaceC_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MCAEPartition partition);


/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MCAEPartition partition, ThreadStateImpl threadState);


}
#endif
