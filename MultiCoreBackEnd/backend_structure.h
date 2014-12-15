#ifndef _H_backend_structure
#define _H_backend_structure

/* structure definitions for built-in object types */

typedef struct {
	int min;
	int max;
} Range;

typedef struct {
	int length;
	Range range;
} Dimension;

typedef struct {
	int beginAt;
	int currentValue;
} Epoch;

/* structure to demarcate the region of a dimension of an array that falls inside a single LPU */

typedef struct {
        Dimension storageDim;
        Dimension partitionDim;
} PartitionDimension;

/* structure for holding a sequence of LPU ids */

typedef struct {
        int startId;
        int endId;
} LpuIdRange;

/* a structure to hold the PPU id of a thread for a space when it is executing stages from it */
typedef struct {
        int id;
        int groupId; // used for higher space computations that each lower space PPU in the
		     // hierarchy executes
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
