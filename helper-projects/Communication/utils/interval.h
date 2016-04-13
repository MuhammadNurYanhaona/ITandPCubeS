#ifndef INTERVAL_H_
#define INTERVAL_H_

/* This header file provides interface specifications for routines that manipulate interval representations of data
 * partition functions. Note that data partition for a single structure within a process translates into a set of,
 * possibly multidimensional, intervals where each logical unit (LPU) multiplexed to the process represents a single
 * interval sequence specification within that set.
*/

//#include <algorithm>
#include <climits>
//#include <cstdlib>
#include <iostream>
//#include <sstream>
#include <stack>
#include <vector>

#include "../structure.h"
#include "list.h"

class IntervalState;

// two constants used for drawing interval sequences
static const char INCLUSION_CHAR = '|';
static const char EXCLUSION_CHAR = '_';

// constant to indicate that the index point returned for an interval sequence is invalid
static const int INVALID_INDEX = INT_MIN;

/* A supporting class to be used to visualize an interval sequence
 * */
class DrawingLine {
private:
	// expanse of the line that which which one or more interval sequences will be drawn
	Dimension dim;
	// the position in the line will be written in periodic interval equal to the label gap
	int labelGap;
	// the characters representing the current status of the line; it spans from dim.min to dim.max and a 1
	// is represented by the inclusion-character and a 0 by the exclusion character
	List<char> *line;
public:
	DrawingLine(Dimension dim, int labelGap);
	void setIndexToOne(int index);
	void draw();
	// this function should be used to reset the line into its original state, which is all 0
	void reset();
};

/* class representing a one-dimensional sequence of intervals occurring at a fixed period for a finite number of times
 * */
class IntervalSeq {
public:
	// the starting point of the first interval in the sequence
	int begin;
	// the length within a period the interval is 1
	int length;
	// the period of the interval sequence
	int period;
	// the number of interval in the sequence
	int count;
public:
	IntervalSeq(int b, int l, int p, int c);
	void increaseCount(int amount) {
		count += amount;
	}
	void draw(DrawingLine *drawingLine);

	/* Note that IT's hierarchical partitioning makes it common for a programmer to partition an already partitioned
	 * data structures into even smaller level partitions. The configuration of the smaller partitions in such cases
	 * assumes that the bigger, higher level data part represent a contiguous portion of indexes in each dimension of
	 * the underlying data structure. That is, however, not true. Thus we apply index transformations inside the code
	 * block. The tree representation of the data contained by a part is also important to determine what needs to be
	 * exchanged between PPUs for shared/overlapped data structure updates. This function is used to transform a sub
	 * interval sequence at a lower level into the actual interval sequence (or sequences) in this level. A repeated
	 * application of this function based on the partitioning scheme used generates the true representation of a part.
	 * */
	List<IntervalSeq*> *transformSubInterval(IntervalSeq *subInterval);

	/* Returns the overlapping region of this interval sequence with another as a form of list of interval sequences;
	 * Returns NULL if the two sequences do not overlap
	 * */
	List<IntervalSeq*> *computeIntersection(IntervalSeq *other, bool logSteps=false);

	// returns the total number of 1's included in the interval sequence
	int getNumOfElements() { return length * count; }

	/* returns the next index in the sequence based on the current state of the interval sequence traversal cursor; it
	 * also updates the state cursor passed as argument; if the end of the sequence has been reached, it resets the
	 * cursor and returns an invalid-index
	 * */
	int getNextIndex(IntervalState *state);

	bool isEqual(IntervalSeq *other);

	// returns true if the interval sequence contains the point; otherwise returns false
	bool contains(int point);

	// two functions for conversion between an interval sequence and its character string representation; the string
	// representation is needed to transfer the interval description over the network
	const char *toString();
	static IntervalSeq *fromString(const char *str);
};

/* A multidimensional sequence of intervals to represent multidimensional data structure parts
 * */
class MultidimensionalIntervalSeq {
protected:
	int dimensionality;
	std::vector<IntervalSeq*> intervals;
public:
	MultidimensionalIntervalSeq(int dimensionality);

	// returns the total number of 1's included in this multidimensional interval sequence
	int getNumOfElements();
	// function to initialize intervals of the current sequence all at once from a template interval vector
	void copyIntervals(std::vector<IntervalSeq*> *templateVector);
	// extends the intersection finding algorithm from above to the multidimensional sequences case
	List<MultidimensionalIntervalSeq*> *computeIntersection(MultidimensionalIntervalSeq *other);

	int getDimensionality() { return dimensionality; }
	void setIntervalForDim(int dimensionNo, IntervalSeq *intervalSeq);
	IntervalSeq *getIntervalForDim(int dimensionNo);
	bool isEqual(MultidimensionalIntervalSeq *other);
	void draw();
	void draw(Dimension dim);

	// This function is used to determine the order of multidimensional sequences in a sorted list; the sequence with
	// component intervals beginning at an earlier point should come before the other. Note that both sequences are
	// assumed to have the same number of dimensions in this comparison. It returns -1 if the current sequence should
	// precede the argument sequence, 1 if the argument sequence should lead, and 0 if their ordering does not matter.
	int compareTo(MultidimensionalIntervalSeq *other);

	// returns true if the interval sequence contains the multidimensional point; otherwise returns false
	bool contains(List<int> *point);

	// function to generate a list of multidimensional interval sequences as a cross-product of lists of one-
	// dimensional interval sequences
	static List<MultidimensionalIntervalSeq*> *generateIntervalSeqs(int dimensionality,
				List<List<IntervalSeq*>*> *intervalSeqLists);

	// functions for conversion between a multidimensional interval sequence and its character string representation;
	// the string representation is needed to transfer the interval description over the network
	const char *toString();
	static MultidimensionalIntervalSeq *fromString(const char *str);

	// these two functions are used when a set of multidimensional interval descriptions are communicated on the wire
	static const char *convertSetToString(List<MultidimensionalIntervalSeq*> *intervals);
	static List<MultidimensionalIntervalSeq*> *constructSetFromString(const char *str);
private:
	// a recursive helper routine to generate multidimensional interval sequences
	static List<MultidimensionalIntervalSeq*> *generateIntervalsFromList(int dimensionality,
				List<List<IntervalSeq*>*> *intervalSeqLists,
				std::vector<IntervalSeq*> *constructionInProgress);
};

/* This is an auxiliary class to be used by multidimensional interval sequence iterator to keep track of the progress
 * along each dimension. That is, it is like a cursor in an interval sequence for dimension.
 * */
class IntervalState {
private:
	int iteration;
	int index;
public:
	// index is initialized to -1 as the cursor will be used as a next pointer that moves the index to 0th position
	// at the beginning of the sequence access
	IntervalState() { iteration = 0; index = -1; }
	inline int getIteration() { return iteration; }
	inline void jumpToNextIteration() { iteration++; index = 0; }
	inline int getIndex() { return index; }
	inline void step() { index++; }
	inline void reset() { iteration = 0; index = -1; }
};

/* an iterator to traverse through a multidimensional interval sequence and get all the indexes included in the sequence
 * in a fixed, incremental order
 * */
class SequenceIterator {
private:
	std::vector<int> *index;
	int dimensionality;
	MultidimensionalIntervalSeq *sequence;
	std::stack<IntervalState*> cursors;
	int elementCount;
	int currentElementNo;
public:
	SequenceIterator(MultidimensionalIntervalSeq *sequence);
	bool hasMoreElements() { return currentElementNo < elementCount; }

	// returns the next index element in the sequence; it returns NULL if there are no more elements and reset its state
	std::vector<int> *getNextElement();

	void reset();
	void printNextElement(std::ostream &stream);
private:
	void initCursorsAndIndex();
};

#endif /* INTERVAL_H_ */
