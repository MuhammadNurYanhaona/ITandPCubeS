#ifndef _H_part_tracking
#define _H_part_tracking

// This is the library that hosts all data structures and mechanism for quickly locating a data structure part given
// its Id.

#include "../utils/list.h"

#include <iostream>
#include <vector>
#include <cstdlib>

class PartIterator;

// the list of parts for a data structure should be stored in an order so that their is a one-to-one correspondence
// between advancement from one LPU to the next with the changes in the data parts that constitute those LPUs. If that
// happens, we can possibly eliminate most of the search associated with finding data parts for LPUs. As a programmer 
// can align dimensions of arrays as he chooses with the dimensions of the LPS they are in, movement from an LPU to 
// the next is not necessarily a movement from a data part to the next data part. Therefore we devise the following 
// class to retain the data-dimension to LPS-dimension alignment information, and store data parts in that order, to 
// counter that problem.  
class DimConfig {
  private:
	int level;
	int dimNo;
  public:
	DimConfig(int level, int dimNo) {
		this->level = level;
		this->dimNo = dimNo;
	}
	inline int getLevel() { return level; }
	inline int getDimNo() { return dimNo; }
};

// super class that serves as a place-holder in a part-container during unique part-id generation process; after the
// container has been set up properly, instances of this class should be replaced with actual data part instances 
class SuperPart {
  protected:
	List<int*> *partId;
  public:
	SuperPart(List<int*> *partId, int dataDimensions);
	List<int*> *getPartId() { return partId; }
	bool isMatchingId(int dimensions, List<int*> *candidatePartId);
};

// This is the base class for holding all the parts corresponding to a single data structure. It only holds the index
// information -- that is the value of a part-Id dimension -- to facilitate parts insertion and searching. 
class PartIdContainer {
  protected:
	int level;
	int dimNo;
	std::vector<int> partArray;
	// a flag to indicate if all ids for current dimension exist within the container; if the answer is YES then 
	// we can directly index into the part or the next container leading to the part instead of doing a search on
	// the part array  
	bool packed;
  public:
	PartIdContainer(DimConfig dimConfig);
	virtual ~PartIdContainer() {}
	// returns the index within the part array where the next container/data-part for a certain part Id may be 
	// found; returns key-not-found if the part-id does not match
	int getCurrentLevelPartIndex(List<int*> *partId);
	// function to be used when we are sure that the part-id exists within the container
	int getCurrentLevelIndexOfExistingPart(List<int*> *partId);
	int getSize() { return partArray.size(); }
	PartIterator *getIterator();
	// This is the interface to be used to locate a data-structure part during task execution (i.e., at individual 
	// LPU construction steps). It first uses the iterator to check if the part can be located at constant time.
	// If that does not work then it does a hierarchical binary search within the container to identify the part. 
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator, int dataDimension);
	// function to be called after all part-Ids are generated; this is used to optimize internal data structures
	// of the container for things such as search performance improvement and freeing unused storage
	virtual void postProcess();
	virtual void print(int indentLevel, std::ostream &stream);
	// two functions to be implemented by sub-classes for container construction and later use for part retrieval
	virtual bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const std::vector<DimConfig> dimOrder, unsigned int position) = 0;
	virtual SuperPart *getPart(List<int*> *partId, PartIterator *iterator) = 0;
	// This is very expensive operation. It constructs and iterator and does a full traversal of the part-container
	// hierarchy to retrieves the part count. Therefore it should be used with great care.
	int getPartCount();
};

// this is the leaf level container that holds the actual parts of a data structure
class PartContainer : public PartIdContainer {
  private:
	std::vector<SuperPart*> dataPartList;
  public:
	PartContainer(DimConfig dimConfig) : PartIdContainer(dimConfig) {}
	~PartContainer();
	bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const std::vector<DimConfig> dimOrder, unsigned int position);
	void postProcess();
	SuperPart *getPartAtIndex(int index) { return dataPartList[index]; }
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator);
	// function to be used to set up an actual part as the replacement for the place-holder SuperPart after a
	// valid data part has been created and initialized based on the nature of the data structure under concern 
	void replacePartAtIndex(SuperPart *repacement, int index);
};

// the part-list-container works as intermediate nodes in the hierarchical part-id-container construction and sub-
// sequent part identification processes
class PartListContainer : public PartIdContainer {
  protected:
	std::vector<PartIdContainer*> nextLevelContainers;
  public:
	PartListContainer(DimConfig dimConfig) : PartIdContainer(dimConfig) {}
	~PartListContainer();
	bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const std::vector<DimConfig> dimOrder, unsigned int position);
	void postProcess();
	void print(int indentLevel, std::ostream &stream);
	PartIdContainer *getNestedContainerAtIndex(int index) { return nextLevelContainers[index]; }
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator);
};

// Note that the part-iterator is an essential component of efficient part identification process. Most of the time
// the part to be returned during an LPU construction should be found in the current or the next location of the
// iterator -- cutting down the search cost altogether. This logic to work, each independent execution unit within
// a segment should maintain its own part iterators for different data structures. This is because different units
// have their own groups of LPUs to execute.
class PartIterator {
  protected:
	std::vector<PartIdContainer*> containerStack;
	std::vector<int> indexStack;
	unsigned int partIdSteps;
  public:
	PartIterator(int partIdSteps);
	SuperPart *getCurrentPart();
	void replaceCurrentPart(SuperPart *replacement);
	void initiate(PartIdContainer *topContainer);
	// move a step ahead in the part-container hierarchy; return false if further forward progress is infeasible
	bool advance() { return advance(partIdSteps - 1); }
	// two functions used during the part-search process to move the iterator to a new location
	void reset();
	void addStep(PartIdContainer *container, int index);
  private:
	bool advance(int lastAccessPoint);
};

#endif
