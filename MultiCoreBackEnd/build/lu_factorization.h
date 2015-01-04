#ifndef _H_luf
#define _H_luf

// for error reporting and diagnostics
#include <iostream>
#include <string>
#include <cstdlib>

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
const int Space_C = 3;
const int Space_D = 4;
const int Space_Count = 5;

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
const int Total_Threads = 32;
const int Threads_Par_Core = 4;

/*-----------------------------------------------------------------------------------
functions for retrieving partition counts in different LPSes
------------------------------------------------------------------------------------*/
int *getLPUsCountOfSpaceB(int ppuCount, Dimension aDim2);
int *getLPUsCountOfSpaceC(int ppuCount, Dimension uDim2);
int *getLPUsCountOfSpaceD(int ppuCount, Dimension uDim1, int s);

/*-----------------------------------------------------------------------------------
functions for getting data ranges along different dimensions of an LPU
-----------------------------------------------------------------------------------*/
PartitionDimension **getaPartForSpaceBLpu(PartitionDimension **aParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getlPartForSpaceBLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getuPartForSpaceBLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getlPartForSpaceCLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getuPartForSpaceCLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId);
PartitionDimension **getlPartForSpaceDLpu(PartitionDimension **lParentLpuDims, 
		int *lpuCount, int *lpuId, int s);
PartitionDimension **getuPartForSpaceDLpu(PartitionDimension **uParentLpuDims, 
		int *lpuCount, int *lpuId, int s);

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

class SpaceC_Content {
  public:
	float *l;
	float *l_column;
	float *u;
};

class SpaceC_LPU : public LPU {
  public:
	float *l;
	PartitionDimension **lPartDims;
	float *l_column;
	PartitionDimension **l_columnPartDims;
	float *u;
	PartitionDimension **uPartDims;
};

class SpaceD_Content {
  public:
	float *l;
	float *u;
};

class SpaceD_LPU : public LPU {
  public:
	float *l;
	PartitionDimension **lPartDims;
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


}
#endif
