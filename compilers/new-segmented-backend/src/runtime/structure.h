#ifndef _H_structure
#define _H_structure

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "../../../common-libs/utils/list.h"

/* default invalid value for any LPU or PPU id */
#define INVALID_ID -1

/* success and failure return types on execution of computation stages */
#define SUCCESS_RUN 1
#define FAILURE_RUN 0

/* class definitions for built-in object types */

class Range {
  public:
	int min;
	int max;
	Range() {
        	min = 0;
        	max = 0;
        }
        Range(int index) {
        	min = index;
        	max = index;
        }
        Range(int min, int max) {
        	this->min = min;
        	this->max = max;
        }
        bool isEqual(Range other) {
        	return (this->min == other.min && this->max == other.max);
        }
};

class Dimension {	
  public:
	Range range;
	int length; // for quick access to length information
	
	int getLength();
	void setLength(int length);
	bool isIncreasing();
	void setLength();

	Range getPositiveRange();
	Range adjustPositiveSubRange(Range positiveSubRange);
	Dimension getNormalizedDimension();
	bool isEqual(Dimension other);
	void print(std::ostream &stream);
};

class Epoch {
  public:	
	int begin;
	int current;
  public:
	Epoch() { begin = 0; current = 0; }
	void advance() { current = current + 1; }
};

/* structure to demarcate the region of a dimension of an array that falls inside a single LPU */

class PartDimension {
  public:
	int count;			// represents the number of partitions made for the underlying dimension
	int index;			// represents the index or Id of the current partition this object refers to 
        Dimension storage;		// actual storage dimension configuration 
        Dimension partition;		// partition configuration
	PartDimension *parent;		// parent partition dimension configuration; if exists

	PartDimension() {
		count = 1;
		index = 0;
		parent = NULL;
	}
	bool isIncluded(int index);	// This function is included only to save time in implementation of the 
					// prototype compiler. Ideally we should not have function calls for checking 
					// if an index is included within a range; rather we should have boolean 
					// expressions directly been applied in underlying context. TODO so we should 
					// avoid using this function in our later optimized implementation.

	int adjustIndex(int index);  	// This is again a time saving implementation of index adjustment when the
					// index under concern is generated from a representative range that starts 
					// from 0 but the original range starts at some non-zero value. We have 
					// such cases when some dimension reordering partition function in some LPS
					// is combined with order preserving partition function in its ancester 
					// LPSes. We can avoid the use of this function in the future by redesigning 
					// the reordering partition functions to take into account non-zero range 
					// beginnings. 
	
	inline int normalizeIndex(int index) {
		return index - partition.range.min;
	}				// Similar to the above, this function is a time saving implementation for
					// shifting index to be relative to the zero based, normalized, beginning 
					// if order preserving partition functions are combined with reordering part-
					// ition functions.

	int safeNormalizeIndex(int index, bool matchToMin); // This function can be used when we are not sure if 
					// the compared index is inside an LPU partition range. When such uncertain
					// transformation is made, we need to ensure that invalid use of the normalized
					// index has not been made. To safeguard against invalid index transformation
					// we should use this function during normalization and specified what value
					// to choose among the min and max as the normalized safety value.

	PartDimension getSubrange(int begin, int end); // This function generate a new part-dimension object that 
					// represent a sub-range of the current object. The storage dimension is copied
					// as it is and the new partition dimension is determined from the arguments. 

	int getDepth();                 // Tells the number of part-dimension objects that forms a path to lead to the
                                        // current part-dimension object  
					
	void print(std::ofstream &stream, int indentLevel);
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
	inline int getPpuId(int lpsId) { return ppuIds[lpsId].id; }
	inline int getPpuCount(int lpsId) { return ppuIds[lpsId].ppuCount; }
	int *getAllPpuCounts();
};

/* a structure to group active LPUs of a dynamic space against corresponding PPUs */
class LPU_Group {
  public:
        int ppuGroupId;
        int *lpuIds;
};

/* base class for LPUs of all LPSes; task specific subclasses will add other necessary fields  */
class LPU {
  public:
        int id;
        bool valid;

        LPU() { id = 0; valid = false; }
        void setId(int id) { this->id = id; }
        void setValidBit(bool valid) { this->valid = valid; }
        bool isValid() { return valid; }
        virtual void print(std::ofstream &stream, int indentLevel) {}
};

#endif
