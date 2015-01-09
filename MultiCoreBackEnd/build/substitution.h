#ifndef _H_s
#define _H_s

// for error reporting and diagnostics
#include <iostream>
#include <string>
#include <cstdlib>

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

namespace s {

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
int *getLPUsCountOfSpaceA(int ppuCount, Dimension mDim2);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getmPartForSpaceALpu(PartitionDimension **mParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getvPartForSpaceALpu(PartitionDimension **vParentLpuDims, 
		int *lpuCount, int *lpuId);

/*-----------------------------------------------------------------------------------
Data structures representing LPS and LPU contents 
------------------------------------------------------------------------------------*/

class SpaceRoot_Content {
  public:
	float *m;
	float *v;
};

class SpaceRoot_LPU : public LPU {
  public:
	float *m;
	PartitionDimension **mPartDims;
	float *v;
	PartitionDimension **vPartDims;
};

class SpaceA_Content {
  public:
	float *m;
	float *v;
};

class SpaceA_LPU : public LPU {
  public:
	float *m;
	PartitionDimension **mPartDims;
	float *v;
	PartitionDimension **vPartDims;
};

/*-----------------------------------------------------------------------------------
Data structures for Array-Metadata and Environment-Links 
------------------------------------------------------------------------------------*/

class ArrayMetadata {
  public:
	Dimension mDims[2];
	Dimension vDims[1];
};
ArrayMetadata arrayMetadata;

class EnvironmentLinks {
  public:
	float *m;
	Dimension mDims[2];
	float *v;
	Dimension vDims[1];
};
EnvironmentLinks environmentLinks;

/*-----------------------------------------------------------------------------------
Data structures for Task-Global and Thread-Local scalar variables
------------------------------------------------------------------------------------*/

class TaskGlobals {
  public:
	Range index_range;
	float v_element;
};

class ThreadLocals {
  public:
	int k;
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
void initializeTask(TaskGlobals taskGlobals, ThreadLocals threadLocals, 
		bool lower_triangular_system);


}
#endif
