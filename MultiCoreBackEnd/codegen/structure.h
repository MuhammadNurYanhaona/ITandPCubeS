#ifndef _H_structure
#define _H_structure

#include <iostream>
#include <fstream>

/* class definitions for built-in object types */

class Range {
  public:
	Range() { min = 0, max = 0; }
	int min;
	int max;
};

class Dimension {
  public:
	Range range;
	int getLength();
	void setLength(int length);
	bool isIncreasing();
	Range getPositiveRange();
	Range adjustPositiveSubRange(Range positiveSubRange);
	Dimension getNormalizedDimension();
	void print(std::ofstream &stream);
};

class Epoch {
  public:
	Epoch() { begin = 0; current = 0; }
	int begin;
	int current;
};

/* structure to demarcate the region of a dimension of an array that falls inside a single LPU */

class PartDimension {
  public:
        Dimension storage;
        Dimension partition;
	void print(std::ofstream &stream);
};

/* structure for holding a sequence of LPU ids */

typedef struct {
        int startId;
        int endId;
} LpuIdRange;

/* a structure to hold the PPU id of a thread for a space when it is executing stages from it */
class PPU_Ids {
  public:
	const char *lpsName;	// a variable useful for printing
        int id;	      		// either the PPU ID or invalid 	
        int groupId;  		// used for higher space computations that each lower space PPU in the
		      		// hierarchy executes; this is set to the PPU ID of the group to which
		      		// current resource belongs to (for resources of the lowest PPS these 
		      		// two ids are always equal)	
	int ppuCount; 
	int groupSize;		// sometimes used for setting up synchronization parameters appropriately
		      		// this represents the total number of leaf level PPUs under the hierarchy
		      		// rooted in PPU of this PPS

	void print(std::ofstream &stream);			
};

/* a structure to hold PPU ids of a thread for different spaces */
class ThreadIds {
  public:
	int threadNo;		// physical thread number
	int lpsCount;		// for determining the number of LPSes
	PPU_Ids *ppuIds;	// IDs for different LPSes

	void print(std::ofstream &stream);
};

/* a structure to group active LPUs of a dynamic space against corresponding PPUs */
class LPU_Group {
  public:
        int ppuGroupId;
        int *lpuIds;
};

/* default invalid value for any LPU or PPU id */
#define INVALID_ID -1

#endif
