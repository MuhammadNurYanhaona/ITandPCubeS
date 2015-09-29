#include "part_distribution.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"
#include "../part-management/part_tracking.h"
#include "../part-management/part_folding.h"
#include <iostream>
#include <vector>

using namespace std;

//--------------------------------------------- LPS Dimension Configuration --------------------------------------------------/

LpsDimConfig::LpsDimConfig() : DimConfig(-1, -1) {
	lpsId = -1;
}

LpsDimConfig::LpsDimConfig(int level, int dimNo, int lpsId) : DimConfig(level, dimNo) {
	this->lpsId = lpsId;
}

bool LpsDimConfig::isEqual(LpsDimConfig other) {
	return this->level == other.level
			&& this->dimNo == other.dimNo
			&& this->lpsId == other.lpsId;
}

void LpsDimConfig::print(int indentLevel, std::ostream &stream) {
	stream << "LPS: " << lpsId << " Level: " << level << " Dimension: " << dimNo << "\n";
}

//------------------------------------------------------- Container ----------------------------------------------------------/

Container::Container(int id, LpsDimConfig config) {
	this->id = id;
	this->config = config;
	this->segmentTags = vector<int>();
	this->parent = NULL;
}

void Container::addSegmentTag(int tag) {
	int existingTagPosition = binsearch::locateKey(segmentTags, tag);
	if (existingTagPosition == KEY_NOT_FOUND) {
		int locationOfInsert = binsearch::locatePointOfInsert(segmentTags, tag);
		segmentTags.insert(segmentTags.begin() + locationOfInsert, tag);
	}
}

void Container::addAllSegmentTags(std::vector<int> tags) {
	for (int i = 0; i < tags.size(); i++) {
		addSegmentTag(tags.at(i));
	}
}

int *Container::getCurrentLevelPartId(int dataDimensions) {
	int *partId = new int[dataDimensions];
	int myLps = config.getLpsId();
	Container *current = this;
	while (current->config.getLpsId() == myLps) {
		partId[current->config.getDimNo()] = current->id;
		current = current->parent;
		if (current == NULL) break;
	}
	return partId;
}

vector<int*> *Container::getPartId(int dataDimensions) {

	vector<int*> *partIdVector = new vector<int*>;
	int *partId = new int[dataDimensions];
	int lastLps = config.getLpsId();
	Container *current = this;

	// level is -1 for the root container; so the recursion should terminate at that point
	while (current->config.getLevel() != -1) {
		int currentContainerId = current->id;
		int currentLps = current->config.getLpsId();
		int currentDimNo = current->config.getDimNo();
		if (currentLps == lastLps) {
			partId[currentDimNo] = currentContainerId;
		} else {
			partIdVector->insert(partIdVector->begin(), partId);
			partId = new int[dataDimensions];
			partId[currentDimNo] = currentContainerId;
		}
		current = current->parent;
		if (current == NULL) break;
	}
	return partIdVector;
}

bool Container::hasSegmentTag(int tag) {
	int location = binsearch::locateKey(segmentTags, tag);
	return (location != KEY_NOT_FOUND);
}

void Container::print(int indentLevel, std::ostream &stream) {
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Container ID: " << id << "\n";
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Segment Tags: ";
	for (unsigned int i = 0; i < segmentTags.size(); i++) {
		stream << segmentTags.at(i) << ", ";
	}
	stream << "\n";
}

PartFolding *Container::foldContainerForSegment(int segmentTag, std::vector<LpsDimConfig> dimOrder, bool foldBack) {
	if (!hasSegmentTag(segmentTag)) return NULL;
	if (foldBack) return foldBackContainer(NULL);
	PartFolding *partFolding = new PartFolding(id, config.getDimNo(), config.getLevel());
	return partFolding;
}

PartFolding *Container::foldBackContainer(PartFolding *foldingUnderConstruct) {

	// level is -1 for the root container that should be skipped during folding
	if (config.getLevel() == -1) return foldingUnderConstruct;

	PartFolding *partFolding = new PartFolding(id, config.getDimNo(), config.getLevel());
	if (foldingUnderConstruct != NULL) {
		partFolding->addDescendant(foldingUnderConstruct);
	}
	return (parent != NULL) ? parent->foldBackContainer(partFolding) : partFolding;
}

//--------------------------------------------------------- Branch -----------------------------------------------------------/

Branch::Branch(LpsDimConfig branchConfig, Container *firstEntry) {
	this->branchConfig = branchConfig;
	descendants = vector<Container*>();
	descendants.push_back(firstEntry);
	descendantIds.push_back(firstEntry->getId());
}

Branch::~Branch() {
	while(descendants.size() > 0) {
		Container *container = descendants.at(descendants.size() - 1);
		descendants.pop_back();
		delete container;
	}
}

void Branch::addEntry(Container *descendant) {
	int key = descendant->getId();
	int location = binsearch::locatePointOfInsert(descendantIds, key);
	descendants.insert(descendants.begin() + location, descendant);
	descendantIds.insert(descendantIds.begin() + location, key);
}

List<Container*> *Branch::getContainerList() {
	List<Container*> *containerList = new List<Container*>;
	for (unsigned int i = 0; i < descendants.size(); i++) {
		containerList->Append(descendants.at(i));
	}
	return containerList;
}

Container *Branch::getEntry(int id) {
	int location = binsearch::locateKey(descendantIds, id);
	if (location != KEY_NOT_FOUND) {
		return descendants.at(location);
	}
	return NULL;
}

List<Container*> *Branch::getContainersForSegment(int segmentTag) {
	List<Container*> *containerList = new List<Container*>;
	for (int i = 0; i < descendants.size(); i++) {
		Container *container = descendants.at(i);
		if (container->hasSegmentTag(segmentTag)) {
			containerList->Append(container);
		}
	}
	return containerList;
}

void Branch::replaceDescendant(Container *descendant) {
	int descendantId = descendant->getId();
	int location = binsearch::locateKey(descendantIds, descendantId);
	descendants.erase(descendants.begin() + location);
	descendants.insert(descendants.begin() + location, descendant);
}

void Branch::print(int indentLevel, std::ostream &stream) {
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Branch Configuration: ";
	branchConfig.print(0, stream);
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Containers on Branch: ";
	for (unsigned int i = 0; i < descendantIds.size(); i++) {
		stream << descendantIds.at(i) << ", ";
	}
	stream << "\n";
	for (unsigned int i = 0; i < descendants.size(); i++) {
		descendants.at(i)->print(indentLevel + 1, stream);
	}
}

//--------------------------------------------------- Branching Container ----------------------------------------------------/

BranchingContainer::~BranchingContainer() {
	while (branches->NumElements() > 0) {
		Branch *branch = branches->Nth(0);
		branches->RemoveAt(0);
		delete branch;
	}
	delete branches;
}

Branch *BranchingContainer::getBranch(int lpsId) {
	for (int i = 0; i < branches->NumElements(); i++) {
		Branch *branch = branches->Nth(i);
		if (branch->getConfig().getLpsId() == lpsId) {
			return branch;
		}
	}
	return NULL;
}

void BranchingContainer::print(int indentLevel, std::ostream &stream) {
	Container::print(indentLevel, stream);
	for (int i = 0; i < branches->NumElements(); i++) {
		branches->Nth(i)->print(indentLevel + 1, stream);
	}
}

void BranchingContainer::insertPart(vector<LpsDimConfig> dimOrder,
		int segmentTag,
		List<int*> *partId, int position) {

	// Note that segment tag has been added to each container along the path to the leaf branch that will represent
	// the would be recorded part-ID so that looking at any position in the part distribution hierarchy, we can
	// immediately say if there is any data part relevant for a particular segment extant under or on that position.
	this->addSegmentTag(segmentTag);

	LpsDimConfig dimConfig = dimOrder.at(position);
	int lpsId = dimConfig.getLpsId();
	int containerId = partId->Nth(dimConfig.getLevel())[dimConfig.getDimNo()];
	bool lastEntry = (position == dimOrder.size() - 1);
	Container *nextContainer = NULL;
	Branch *branch = getBranch(lpsId);
	if (branch != NULL) {
		nextContainer = branch->getEntry(containerId);
	}
	if (nextContainer == NULL) {
		nextContainer = (lastEntry) ? new Container(containerId, dimConfig)
				: new BranchingContainer(containerId, dimConfig);
		nextContainer->addSegmentTag(segmentTag);
		if (branch == NULL) {
			branches->Append(new Branch(dimConfig, nextContainer));
		} else {
			branch->addEntry(nextContainer);
		}
	} else {
		// A hybrid container is needed when one or more intermediate steps for a hierarchical data partition contain
		// their own larger data parts for computation. To clarify, suppose Space A divides Space B and there are
		// computations occurring in both spaces and data is reordered by Space A after Space B. Then two separate
		// allocations will be maintained for larger Space B data parts and smaller Space A sub-data parts. Therefore,
		// the part distribution tree will contain hybrid containers marking the super parts belonging to Space B.
		BranchingContainer *intermediate = dynamic_cast<BranchingContainer*>(nextContainer);
		HybridBranchingContainer *hybrid = dynamic_cast<HybridBranchingContainer*>(nextContainer);
		if (lastEntry && intermediate != NULL && hybrid == NULL) {
			hybrid = HybridBranchingContainer::convertIntermediate(intermediate, segmentTag);
			branch->replaceDescendant(hybrid);
			nextContainer = hybrid;
		} else if (!lastEntry && intermediate == NULL) {
			hybrid = HybridBranchingContainer::convertLeaf(nextContainer, segmentTag);
			branch->replaceDescendant(hybrid);
			nextContainer = hybrid;
		} else if (hybrid != NULL) {
			hybrid->addSegmentTag(segmentTag, lastEntry);
		} else {
			nextContainer->addSegmentTag(segmentTag);
		}
	}
	nextContainer->setParent(this);
	if (!lastEntry) {
		BranchingContainer *nextLevel = reinterpret_cast<BranchingContainer*>(nextContainer);
		nextLevel->insertPart(dimOrder, segmentTag, partId, position + 1);
	}
}

Container *BranchingContainer::getContainer(List<int*> *pathToContainer, vector<LpsDimConfig> dimOrder, int position) {
	LpsDimConfig dimConfig = dimOrder.at(position);
	int lpsId = dimConfig.getLpsId();
	int containerId = pathToContainer->Nth(dimConfig.getLevel())[dimConfig.getDimNo()];
	bool lastEntry = (position == dimOrder.size() - 1);
	Branch *branch = getBranch(lpsId);
	if (branch == NULL) return NULL;
	Container *container = branch->getEntry(containerId);
	if (lastEntry || container == NULL) return container;
	BranchingContainer *nextLevel = reinterpret_cast<BranchingContainer*>(container);
	return nextLevel->getContainer(pathToContainer, dimOrder, position + 1);
}

List<Container*> *BranchingContainer::listDescendantContainersForLps(int lpsId, int segmentTag, bool segmentSpecific) {

	List<Container*> *containerList = new List<Container*>;
	Branch *branch = getBranch(lpsId);
	if (branch == NULL) {
		return containerList;
	}

	List<Container*> *containersOnBranch = NULL;
	if (segmentSpecific) {
		containersOnBranch = branch->getContainersForSegment(segmentTag);
	} else {
		containersOnBranch = branch->getContainerList();
	}

	for (int i = 0; i < containersOnBranch->NumElements(); i++) {
		Container *nextContainer = containersOnBranch->Nth(i);
		BranchingContainer *nextBranch = dynamic_cast<BranchingContainer*>(nextContainer);
		if (nextBranch == NULL || nextBranch->getBranch(lpsId) == NULL) {
			containerList->Append(nextContainer);
		} else {
			List<Container*> *nextBranchList =
					nextBranch->listDescendantContainersForLps(lpsId, segmentTag, segmentSpecific);
			containerList->AppendAll(nextBranchList);
			delete nextBranchList;
		}
	}
	delete containersOnBranch;
	return containerList;
}

PartFolding *BranchingContainer::foldContainerForSegment(int segmentTag,
		vector<LpsDimConfig> dimOrder,
		bool foldBack) {

	if (!hasSegmentTag(segmentTag)) return NULL;

	// If the configuration level is -1 then this container is the root of the part-distribution-tree and we need to skip it
	// as the dimension-order vector derived from the data-partition-configuration does not consider a distribution root that
	// holds all data parts.
	int position = 0;
	if (config.getLevel() != -1) {
		while (!dimOrder.at(position).isEqual(config)) position++;
	} else position = -1;

	int lastDimOrderEntry = dimOrder.size() - 1;
	if (position == lastDimOrderEntry) {
		HybridBranchingContainer *hybrid = dynamic_cast<HybridBranchingContainer*>(this);
		if (hybrid != NULL) {
			Container *leafContainer = hybrid->getLeaf();
			return leafContainer->foldContainerForSegment(segmentTag, dimOrder, foldBack);
		} else {
			return Container::foldContainerForSegment(segmentTag, dimOrder, foldBack);
		}
	}

	PartFolding *folding = new PartFolding(id, config.getDimNo(), config.getLevel());
	foldContainer(segmentTag, folding->getDescendants(), dimOrder, position + 1);

	if (folding->getDescendants()->NumElements() == 0) {
		delete folding;
		return NULL;
	}
	return (foldBack && parent != NULL) ? parent->foldBackContainer(folding) : folding;
}

void BranchingContainer::foldContainer(int segmentTag, List<PartFolding*> *fold,
		vector<LpsDimConfig> dimOrder, int position) {

	LpsDimConfig nextConfig = dimOrder.at(position);
	Branch *branch = getBranch(nextConfig.getLpsId());
	if (branch != NULL) {

		List<Container*> *containerList = branch->getContainersForSegment(segmentTag);
		int nextPosition = position + 1;
		for (int i = 0; i < containerList->NumElements(); i++) {
			Container *container = containerList->Nth(i);
			BranchingContainer *nextBranch = dynamic_cast<BranchingContainer*>(container);
			HybridBranchingContainer *hybrid = dynamic_cast<HybridBranchingContainer*>(container);
			PartFolding *foldElement = NULL;

			int lastEntry = dimOrder.size() - 1;
			if (nextPosition < lastEntry) {
				PartFolding *subFold = new PartFolding(container->getId(), nextConfig.getDimNo(), nextConfig.getLevel());
				nextBranch->foldContainer(segmentTag, subFold->getDescendants(), dimOrder, nextPosition);
				if (subFold->getDescendants()->NumElements() > 0) {
					foldElement = subFold;
				} else delete subFold;
			} else {
				container = (hybrid != NULL) ? hybrid->getLeaf() : container;
				foldElement = container->foldContainerForSegment(segmentTag, dimOrder, false);
			}

			if (foldElement == NULL) continue;

			// if this is the first sub-fold then we add it in the list right away
			if (fold->NumElements() == 0) {
				fold->Append(foldElement);
			// otherwise we first check if we can coalesce the current sub-fold with the previous to make the representation
			// more compact
			} else {
				PartFolding *previousElement = fold->Nth(fold->NumElements() - 1);
				int containerId = container->getId();
				if (previousElement->getIdRange().max == containerId - 1
						&& foldElement->isEqualInContent(previousElement)) {
					previousElement->coalesce(Range(containerId, containerId));
					delete foldElement;
				} else {
					fold->Append(foldElement);
				}
			}
		}
	}
}

//----------------------------------------------- Hybrid Branching Container -------------------------------------------------/

HybridBranchingContainer::HybridBranchingContainer(BranchingContainer *branch, Container *leaf)
	: BranchingContainer(branch->getId(), branch->getConfig()) {
	this->leaf = leaf;
	this->segmentTags = branch->getSegmentTags();
	this->parent = branch->getParent();
	this->branches = branch->getBranches();
}

void HybridBranchingContainer::print(int indentLevel, std::ostream &stream) {
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Hybrid Container:\n";
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Leaf Configuration:\n";
	leaf->print(indentLevel + 1, stream);
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << "Intermediate Configuration:\n";
	BranchingContainer::print(indentLevel + 1, stream);
}

HybridBranchingContainer *HybridBranchingContainer::convertLeaf(Container *leafContainer, int branchSegmentTag) {

	LpsDimConfig leafConfig = leafContainer->getConfig();
	int leafId = leafContainer->getId();
	BranchingContainer *intermediate = new BranchingContainer(leafId, leafConfig);
	intermediate->addSegmentTag(branchSegmentTag);

	// Note that all segment tags from the leaf container have been inserted in the intermediate container but the
	// converse is not done in the next function. This is because, a leaf container lies within the hybrid but the
	// intermediate part of the hybrid works as a normal branching container, exposed to the hierarchy. When a search
	// for a leaf container with a particular segment Id has been issued, we should be able to locate the hybrid that
	// may contain it. If we do not copy segment tags of the leaf in the branch container then we may miss valid leaf
	// containers residing within hybrid containers.
	intermediate->addAllSegmentTags(leafContainer->getSegmentTags());

	return new HybridBranchingContainer(intermediate, leafContainer);
}

HybridBranchingContainer *HybridBranchingContainer::convertIntermediate(BranchingContainer *branchContainer,
		int terminalSegmentTag) {
	LpsDimConfig intermediateConfig = branchContainer->getConfig();
	int intermediateId = branchContainer->getId();
	Container *leaf = new Container(intermediateId, intermediateConfig);
	leaf->setParent(branchContainer->getParent());
	leaf->addSegmentTag(terminalSegmentTag);
	return new HybridBranchingContainer(branchContainer, leaf);
}

void HybridBranchingContainer::addSegmentTag(int segmentTag, bool leafLevelTag) {
	if (leafLevelTag) leaf->addSegmentTag(segmentTag);
	Container::addSegmentTag(segmentTag);
}
