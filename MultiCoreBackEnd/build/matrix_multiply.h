#ifndef _H_mm
#define _H_mm

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

namespace mm {

/*-----------------------------------------------------------------------------------
constants for LPSes
------------------------------------------------------------------------------------*/
const int Space_Root = 0;
const int Space_A = 1;
const int Space_Count = 2;

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
int *getLPUsCountOfSpaceA(int ppuCount, Dimension cDim1, int k, Dimension cDim2, int l);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getaPartForSpaceALpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId, int k);
PartitionDimension **getbPartForSpaceALpu(PartitionDimension **bParentLpuDims, 
		int *lpuCount, int *lpuId, int l);
PartitionDimension **getcPartForSpaceALpu(PartitionDimension **cParentLpuDims, 
		int *lpuCount, int *lpuId, int k, int l);

/*-----------------------------------------------------------------------------------
Data structures representing LPS and LPU contents 
------------------------------------------------------------------------------------*/

class SpaceRoot_Content {
  public:
	float *a;
	float *b;
	float *c;
};

class SpaceRoot_LPU : public LPU {
  public:
	float *a;
	PartitionDimension **aPartDims;
	float *b;
	PartitionDimension **bPartDims;
	float *c;
	PartitionDimension **cPartDims;
};

class SpaceA_Content {
  public:
	float *a;
	float *b;
	float *c;
};

class SpaceA_LPU : public LPU {
  public:
	float *a;
	PartitionDimension **aPartDims;
	float *b;
	PartitionDimension **bPartDims;
	float *c;
	PartitionDimension **cPartDims;
	int lpuId[2];
};

/*-----------------------------------------------------------------------------------
Data structures for Array-Metadata and Environment-Links 
------------------------------------------------------------------------------------*/

class ArrayMetadata {
  public:
	Dimension aDims[2];
	Dimension bDims[2];
	Dimension cDims[2];
};
ArrayMetadata arrayMetadata;

class EnvironmentLinks {
  public:
	float *a;
	Dimension aDims[2];
	float *b;
	Dimension bDims[2];
};
EnvironmentLinks environmentLinks;

/*-----------------------------------------------------------------------------------
Data structures for Task-Global and Thread-Local scalar variables
------------------------------------------------------------------------------------*/

class TaskGlobals {
  public:
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
void initializeTask(ArrayMetadata arrayMetadata, 
		EnvironmentLinks environmentLinks, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MMPartition partition);

/*-----------------------------------------------------------------------------------
functions for compute stages 
------------------------------------------------------------------------------------*/

void mm_function0(SpaceA_LPU lpu, 
		ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, MMPartition partition);


/*-----------------------------------------------------------------------------------
The run method for thread simulating the task flow 
------------------------------------------------------------------------------------*/

void run(ArrayMetadata arrayMetadata, 
		TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		MMPartition partition, ThreadStateImpl threadState);


}
#endif
