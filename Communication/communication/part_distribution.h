#ifndef PART_DISTRIBUTION_H_
#define PART_DISTRIBUTION_H_

/* This header file contains classes and features that are similar to that of part_tracking library. The difference is
 * that here the classes are for tracking parts of multiple segments. Furthermore, here we construct a single part-hierarchy
 * per data structure that maintain information regarding how that data structure has been partitioned in various ways and
 * which part went to which segment. This is done so that we can determine the communication/synchronization need on an
 * update of a data structure. By investigating the branches of a multi-branched, multi-segmented part hierarchy we should
 * be able to say what data needs to be exchanged between a pair of communicating segments (also, within a segment, between
 * overlapped data parts).
 * */

#include "../utils/list.h"
#include "../part-management/part_tracking.h"
#include "../part-management/part_folding.h"
#include <vector>
#include <cstdlib>

/* The dim-configuration class used to tag each part-container level in the part-tracking hierarchy is not sufficient when
 * we have a graph combining parts from independent hierarchies. So this extension is provided to include the LPS ID along
 * with the level and dimension numbers used earlier.
 * */
class LpsDimConfig {
protected:
	int level;
	int dimNo;
	int lpsId;
public:
	LpsDimConfig() {
		level = dimNo = lpsId = -1;
	}
	LpsDimConfig(int level, int dimNo, int lpsId) {
		this->level = level;
		this->dimNo = dimNo;
		this->lpsId = lpsId;
	}
	inline int getLevel() { return level; }
	inline int getDimNo() { return dimNo; }
	inline int getLpsId() { return lpsId; }
	bool isEqual(LpsDimConfig other) {
		return this->level == other.level
				&& this->dimNo == other.dimNo
				&& this->lpsId == other.lpsId;
	}
};

/* A container represents a leaf level entry or a part that may belong to one or more segments. The part ID at the position
 * indicated by the LPS-dimension-configuration is the same as the container ID. The segment-tags represents all segments
 * that share this container. Note that later this leaf container has been extended into Branching-Container to represent
 * intermediate locations in the part hierarchy. It is important to understand that for a branching container two segments
 * sharing it does not mean they share everything underneath. Rather it means that they have bifurcated from a common container.
 * Having the segment tags like this helps in identifying the potential participants in a communication at any level of the
 * multi-branching hierarchy.
 * */
class Container {
protected:
	int id;
	std::vector<int> segmentTags;
	LpsDimConfig config;
	// a parent pointer is maintained so that we can determine the the chain of Ids that lead to the current container
	Container *parent;
public:
	Container(int id, LpsDimConfig config);
	virtual ~Container() {}
	std::vector<int> getSegmentTags() { return segmentTags; }
	void setParent(Container *parent) { this->parent = parent; }
	Container *getParent() { return parent; }
	void addSegmentTag(int tag);
	void addAllSegmentTags(std::vector<int> tags);
	int getId() { return id; }
	LpsDimConfig getConfig() { return config; }
	bool hasSegmentTag(int tag);

	// recreates the (possibly multidimensional) hierarchical ID of the part represented by this container
	std::vector<int*> *getPartId(int dataDimensions);
	// generates the part Id for the LPS level this container resides within
	int *getCurrentLevelPartId(int dataDimensions);

	// This function is mainly needed for intermediate BranchingContainers. It has been added here to have a common interface
	// to retrieve compact description of leaf (already compact) and intermediate containers in the same way.
	virtual PartFolding *foldContainerForSegment(int segmentId, std::vector<LpsDimConfig> dimOrder, bool foldBack);

	// a helper function for foldContainerForSegment recursion
	PartFolding *foldBackContainer(PartFolding *foldingUnderConstruct = NULL);
};

/* As the name suggests, this class stands for a strand (i.e. an LPS) on a possibly multi-branching point of the part hierarchy.
 * */
class Branch {
protected:
	// a branch configuration is needed to guide any search/traversal process to appropriate branch from the part-container
	// root that is holding this branch, and possibly other branches.
	LpsDimConfig branchConfig;
	// list of next level containers along a particular LPS branch
	std::vector<Container*> descendants;
	// the IDs of the descendants are maintained in an ordered manner to quickly search and identify a particular descendant
	std::vector<int> descendentIds;
public:
	Branch(LpsDimConfig branchConfig, Container *firstEntry);
	~Branch();
	LpsDimConfig getConfig() { return branchConfig; }
	void addEntry(Container *descendant);
	// returns the container with an specific Id on the branch if exists; otherwise returns NULL
	Container *getEntry(int id);
	// returns all containers that have a particular segment tag
	List<Container*> *getContainersForSegment(int segmentTag);
	// replaces an already existing descendant container with a new version of it
	void replaceDescendant(Container *descendant);
};

/* This represents an intermediate container in the part-container hierarchy
 * */
class BranchingContainer : public Container {
protected:
	List<Branch*> *branches;
public:
	BranchingContainer(int id, LpsDimConfig config): Container(id, config) {
		branches = new List<Branch*>;
	}
	virtual ~BranchingContainer();
	Branch *getBranch(int lpsId);
	List<Branch*> *getBranches() { return branches;}

	// This is the interface to use to populate the branching hierarchy for a data parts; it creates new branches as needed,
	// deposits new segment token in already created containers along a part's path, and create a new leaf level container to
	// register a part when that occurs for the first time. The final parameter is used to keep track of progress of the
	// recursive routine and should not be specified when called from outside.
	void insertPart(std::vector<LpsDimConfig> dimOrder, int segmentTag, List<int*> *partId, int position = 0);

	// This returns any leaf level container representing a particular part or any intermediate container depending on the
	// specification of the dimOrder and the path Id chain passed as the argument. Just like the previous case, the last
	// parameter is used to track the progress of the recursive function and the outside caller should not use it.
	Container *getContainer(List<int*> *containerPath, std::vector<LpsDimConfig> dimOrder, int position = 0);

	// This returns all the descendant containers that belongs to a particular segment (indicated by the segment tag) that are
	// below the sub-tree rooted at the current container. Note that for a multi-dimensional LPS, the containers here are all
	// that are at the lowest level in the part hierarchy. To explain this with an example, consider the current container is
	// at Space A and a 2D Space B divides A. Then the part hierarchy rooted at the current container may look like as follows
	//	current Container
	// 	|----Space-B:dim-1 [a set of containers]
	// 		|-----Space-B:dim-2 [a set of containers under each container in the above]
	// The container hierarchy has this shape because each dimension is treated separately. Now the following function will
	// return all Space-B:dim-2 containers underneath the current container that has a particular segment tag.
	// Note that this function does not recursively goes down to find the container for the specified LPS, i.e., the lpsId
	// should match one of the branches of the current container.
	List<Container*> *listDescendantContainersForLps(int lpsId, int segmentTag);

	// This forms a compact data representation for the data parts for a particular segment that fall within the confinement of
	// the sub-tree rooted at the current container. Note that if the dimOrder vector starts from a dimension and level higher
	// than the current container then entries of the vector will be skipped until the configuration for the current container
	// matches with some entry. The fold that will be generated should have the configuration of the upper levels nonetheless.
	// This is because to generate the interval description of data residing within the confinement of the current container we
	// need to know the configuration of the confinement itself first. Therefore the last parameter should be set to true when
	// calling this function except in unusual circumstances.
	PartFolding *foldContainerForSegment(int segmentTag, std::vector<LpsDimConfig> dimOrder, bool foldBack);
protected:
	// a recursive helper routine for the fold-container-for-segment function
	void foldContainer(int segmentId, List<PartFolding*> *fold, std::vector<LpsDimConfig> dimOrder, int position = 0);
};

/* Sometimes some intermediate containers may themselves represent a higher level LPS data part along with a holder of further
 * lower level data parts. For example, if Space A is an ancestor of Space B and a data structure x has been used in both spaces
 * and further there is one or more reordering of x's dimension indexes in between Space A and B partitions then we will have
 * separate memory allocations for Space A and Space B data parts for x. In that scenario, a Space A part will be a superset of
 * several Space B parts. For this kind of scenario, an intermediate branching-container for Space A leading to Space B lower
 * level containers itself a leaf level container. To be able to record this instances properly, we extend the branching-container
 * in this class.
 * */
class HybridBranchingContainer : public BranchingContainer {
protected:
	// the container to represent leaf level instance a.k.a a part
	Container *leaf;
public:
	HybridBranchingContainer(BranchingContainer *branch, Container *leaf)
		: BranchingContainer(branch->getId(), branch->getConfig()) {
		this->leaf = leaf;
		this->segmentTags = branch->getSegmentTags();
		this->parent = branch->getParent();
		this->branches = branch->getBranches();
	}

	// During the part hierarchy construction process a hybrid-branching-container will always be created from an existing leaf
	// level container or an intermediate branching container. So two static functions have been provided for the conversion and
	// the original constructor has been kept hidden.
	static HybridBranchingContainer *convertLeaf(Container *leafContainer, int branchSegmentTag);
	static HybridBranchingContainer *convertIntermediate(BranchingContainer *branchContainer, int terminalSegmentTag);

	Container *getLeaf() { return leaf; }
	void addSegmentTag(int segmentTag, bool leafLevelTag);
};

#endif /* PART_DISTRIBUTION_H_ */
