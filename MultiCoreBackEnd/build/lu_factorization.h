#ifndef _H_luf
#define _H_luf

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

namespace luf {

/*-----------------------------------------------------------------------------------
constants for LPSes
------------------------------------------------------------------------------------*/
const int Space_Root = 0;
const int Space_A = 1;
const int Space_B = 2;
const int Space_Count = 3;

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
const int Total_Threads = 16;
const int Threads_Par_Core = 1;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/
int *getLPUsCountOfSpaceB(int ppuCount, Dimension aDim2);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getaPartForSpaceBLpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getlPartForSpaceBLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getuPartForSpaceBLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId);

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
ArrayMetadata arrayMetadata;

class EnvironmentLinks {
  public:
	float *a;
	Dimension aDims[2];
};
EnvironmentLinks environmentLinks;

/*-----------------------------------------------------------------------------------
Data structures for Task-Global and Thread-Local scalar variables
------------------------------------------------------------------------------------*/

class TaskGlobals {
  public:
	int pivot;
};

class ThreadLocals {
  public:
	int k;
	Epoch t;
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
		LUFPartition partition);

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void Prepare(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);

void Select_Pivot(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);

void Store_Pivot(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);

void Interchange_Rows(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);

void Update_Lower(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);

void Update_Upper(SpaceB_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, LUFPartition partition);


/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		LUFPartition partition, ThreadStateImpl threadState);


}
#endif
