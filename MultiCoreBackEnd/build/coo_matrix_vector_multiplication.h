#ifndef _H_coomvm
#define _H_coomvm

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

namespace coomvm {

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
int *getLPUsCountOfSpaceA(int ppuCount, Dimension w_localDim1);
int *getLPUsCountOfSpaceB(int ppuCount, Dimension wDim1, int r);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getmPartForSpaceALpu(PartitionDimension **mParentLpuDims, 
		int *lpuCount, int *lpuId, int p);
PartitionDimension **getw_localPartForSpaceALpu(PartitionDimension **w_localParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getwPartForSpaceBLpu(PartitionDimension **wParentLpuDims, 
		int *lpuCount, int *lpuId, int r);
PartitionDimension **getw_localPartForSpaceBLpu(PartitionDimension **w_localParentLpuDims, 
		int *lpuCount, int *lpuId, int r);

/*-----------------------------------------------------------------------------------
Data structures representing LPS and LPU contents 
------------------------------------------------------------------------------------*/

class SpaceRoot_Content {
  public:
	ValueCoordinatePair *m;
	float *v;
	float *w;
	float *w_local;
};

class SpaceRoot_LPU : public LPU {
  public:
	ValueCoordinatePair *m;
	PartitionDimension **mPartDims;
	float *v;
	PartitionDimension **vPartDims;
	float *w;
	PartitionDimension **wPartDims;
	float *w_local;
	PartitionDimension **w_localPartDims;
};

class SpaceA_Content {
  public:
	ValueCoordinatePair *m;
	float *v;
	float *w_local;
};

class SpaceA_LPU : public LPU {
  public:
	ValueCoordinatePair *m;
	PartitionDimension **mPartDims;
	float *v;
	PartitionDimension **vPartDims;
	float *w_local;
	PartitionDimension **w_localPartDims;
};

class SpaceB_Content {
  public:
	float *w;
	float *w_local;
};

class SpaceB_LPU : public LPU {
  public:
	float *w;
	PartitionDimension **wPartDims;
	float *w_local;
	PartitionDimension **w_localPartDims;
};

/*-----------------------------------------------------------------------------------
Data structures for Array-Metadata and Environment-Links 
------------------------------------------------------------------------------------*/

class ArrayMetadata {
  public:
	Dimension mDims[1];
	Dimension vDims[1];
	Dimension wDims[1];
	Dimension w_localDims[2];
};
ArrayMetadata arrayMetadata;

class EnvironmentLinks {
  public:
	ValueCoordinatePair *m;
	Dimension mDims[1];
	float *v;
	Dimension vDims[1];
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
void initializeTask(TaskGlobals taskGlobals, 
		ThreadLocals threadLocals, 
		COOMVMPartition partition);


}
#endif
