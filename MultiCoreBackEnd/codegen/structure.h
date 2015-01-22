#ifndef _H_structure
#define _H_structure

/* class definitions for built-in object types */

class Range {
  public:
	int min;
	int max;
};

class Dimension {
  public:
	int length;
	Range range;
};

class Epoch {
  public:
	int begin;
	int current;
};

/* structure to demarcate the region of a dimension of an array that falls inside a single LPU */

typedef struct {
        Dimension *storageDim;
        Dimension *partitionDim;
} PartitionDimension;

/* structure for holding a sequence of LPU ids */

typedef struct {
        int startId;
        int endId;
} LpuIdRange;

/* a structure to hold the PPU id of a thread for a space when it is executing stages from it */
typedef struct {
        int id;	      // either the PPU ID or invalid 	
        int groupId;  // used for higher space computations that each lower space PPU in the
		      // hierarchy executes; this is set to the PPU ID of the group to which
		      // current resource belongs to (for resources of the lowest PPS these 
		      // two ids are always equal)	
	int ppuCount; 
	int groupSize;// sometimes used for setting up synchronization parameters appropriately
		      // this represents the total number of leaf level PPUs under the hierarchy
		      // rooted in PPU of this PPS		
} PPU_Ids;

/* a structure to hold PPU ids of a thread for different spaces */
typedef struct {
	PPU_Ids *ppuIds;
} ThreadIds;

/* a structure to group active LPUs of a dynamic space against corresponding PPUs */
typedef struct {
        int ppuGroupId;
        int *lpuIds;
} LPU_Group;

/* default invalid value for any LPU or PPU id */
#define INVALID_ID -1

#endif
