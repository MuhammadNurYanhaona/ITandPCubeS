#ifndef _H_interval
#define _H_interval

/* This header file provides interface specifications for routines that manipulate interval representations
   of data partition functions. Note that data partition for a single structure within a process translates
   into a set of, possibly multi-dimensional, intervals where each logical unit (LPU) multiplexed to the 
   process represents a single interval sequence specification within that set.

   @authors: Andrew Grimshaw & Muhammad Yanhaona	 
*/

#include "list.h"

// This class is used to represents the spread of a single dimension of an array. As IT arrays can start from
// an arbitrary index, we need both start and end to be specified.
class Line {
  protected:
	int start;
	int end;
	int length;
  public:
	Line(int end) {
		this->start = 0;
		this->end = end;
		this->length = end + 1;
	}
	Line (int start, int end) {
		this->start = start;
		this->end = end;
		this->length = start - end + 1;
	}
	bool isEqual(Line *other) { return (this->start == other->start) && (this->end == other->end); }
	inline int getStart() { return start;}
	inline int getEnd() { return end; }
	inline int getLength() { return length; }
};

// A line interval provides a four-tuple specification of a uni-dimensional interval sequence of the form 
// <s, l, g, n>. Here,
// s:begin is the starting point of the first interval in the sequence
// l:length is the length of each interval in the sequence
// g:gap is the gap between successive intervals' beginnings if the sequence has more than one intervals
// n:count is the total number of intervals
// Note that the last interval in a sequence may be shorter than l if part of its fall outside the end of 
// the dimension line. To be able to take that into account in any calculation, a reference to the dimension 
// line has been added in the class.   
class LineInterval {
  protected:
	Line *line;
	int begin;
	int length;
	int count;
	int gap;
  public:
	LineInterval(int begin, int length, int count, int gap);
	void setLine(Line *line) { this->line = line; }
	inline Line *getLine() { return line; }
	inline int getBegin() { return begin; }
	inline int getLength() { return length; }
	inline int getCount() { return count; }
	inline int getGap() { return gap; }
	bool isEqual(LineInterval *other);
	
	// Two line interval specifications can be joined together to form a single interval specification
	// only if they have the same interval count and their i'th intervals are adjacent for all i. This
	// implies that they have the same gap too. This function is given so that the number of interval
	// specifications can be kept small. If the two sequences cannot be joined then the method returns 
	// null. 
	LineInterval *join(LineInterval *other);

	// Returns a line in the form of an interval specification
	static LineInterval *getFullLineInterval(Line *line);

	// TODO implement the following three methods
	// Returns the actual index of the nth element in the interval sequence. For example, if an interval 
	// specification is like <0, 1, 10, 5> then getNthElement(2) should return 10. If there is no nth 
	// element then it should return INT_MIN of <climits> library to indicate an invalid request.
	int getNthElement(int n);
	// Returns the total number of elements of the dimension line that fall within the interval sequence
	// represented by current specification
	int getTotalElements();	
	// decides if two interval sequences overlap anywhere
	bool doesOverlap(LineInterval *other);
};

// A hyperplane interval specification defines a multidimensional interval sequence where configurations for 
// individual dimensions are listed in the form of line-intervals.   
class HyperplaneInterval {
  protected:
	// the dimensionality of the hyperplane
	int dimensions;
	// the list of line-intervals for the multidimensional interval sequence
	List<LineInterval*> *lineIntervals;
  public:
	HyperplaneInterval(int dimensions, List<LineInterval*> *lineIntervals) {
		this->dimensions = dimensions;
		this->lineIntervals = lineIntervals;
	}
	inline int getDimensions() { return dimensions; }
	inline List<LineInterval*> *getLineIntervals() { return lineIntervals; }
	inline LineInterval *getLineInterval(int dimensionNo) { return lineIntervals->Nth(dimensionNo); }
	bool isEqual(HyperplaneInterval *other);

	// A pair of N dimensional interval sequences can be joined to form a composite interval if they are
	// the same along N - 1 dimensions and can be joined along the remaining one.
	HyperplaneInterval *join(HyperplaneInterval *other);
	
	// TODO implement the following three methods
	// Returns the n-th element of a multidimensional interval sequence. The element returned is a point 
	// in in the hyperplane and represented by its index along different dimensions; i.e., a tuple. Just 
	// like for the linear interval sequence, if there are less than n elements then it should return 
	// INT_MIN as an invalid  request indicator. Any fixed layout of the elements of the multidimensional 
	// sequence is exceptable.      
	int *getNthElement(int n);
	// returns the total number elements this multidimensional interval sequence posesses
	int getTotalElements();
	// decides if two interval sequences overlap anywhere
	bool doesOverlap(HyperplaneInterval *other);
};

// A helper collection class built over the list data structure to hold a set of unique interval sequences. It
// is assumed that all interval sequences in a set has the same dimensionality and do not overlap
class IntervalSet {
  protected:
	List<HyperplaneInterval*> *intervals;
  public:
	IntervalSet() { intervals = new List<HyperplaneInterval*>; }
	// checks if the argument interval is within the set
	bool contains(HyperplaneInterval *interval);
	// adds the argument interval if it is not already in the set
	void add(HyperplaneInterval *interval);
	// removes the argument interval from the set if it exists; otherwise does nothing
	void remove(HyperplaneInterval *interval);
	// remove all intervals from the set; thereby empties it
	void clear();
	inline List<HyperplaneInterval*> *getIntervalList() { return intervals; }

	// TODO implement the following	three methods
	// This computes the union of current interval set and the argument set. Note that the union computation
	// is not a mere accumulation of unique interval sequences from both sets. Overlappings among intervals
	// need to be taken into consideration and should be removed so that the new set has no element appearing
	// more than once. Finally, note that the returned set is a new interval set instance. The current set
	// remains unchanged after the function call. 
	IntervalSet *getUnion(IntervalSet *other);
	// This computes the intersection of current interval set with the argument interval set. If the two sets
	// do not overlap then it should return NULL
	IntervalSet *getIntersection(IntervalSet *other);
	// Subtract the interval sequences found in the argument interval set from the current one. If the two 
	// sets are the same then return NULL
	IntervalSet *getSubtraction(IntervalSet *other);
};

#endif
