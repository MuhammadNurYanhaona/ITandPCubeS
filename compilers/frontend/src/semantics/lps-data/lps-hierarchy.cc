#include "../task_space.h"
#include "../symbol.h"
#include "../partition_function.h"
#include "../../common/constant.h"
#include "../../common/location.h"
#include "../../common/errors.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_def.h"
#include "../../static-analysis/usage_statistic.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <deque>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------- Partition Hierarchy -------------------------------------------------/

PartitionHierarchy::PartitionHierarchy() {
	spaceHierarchy = new Hashtable<Space*>;
}

Space *PartitionHierarchy::getSpace(char spaceId) {
	char *key = (char *) malloc(sizeof(char) * 2);
	key[0] = spaceId;
	key[1] = '\0';
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getSubspace(char spaceId) {
	int suffixLength = strlen(Space::SubSpaceSuffix);
	char *key = (char *) malloc(sizeof(char) * (suffixLength + 2));
	key[0] = spaceId;
	key[1] = '\0';
	strcat(key, Space::SubSpaceSuffix);
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getRootSpace() {
	return spaceHierarchy->Lookup(Space::RootSpaceName);
}

bool PartitionHierarchy::addNewSpace(Space *space) {
	Space *duplicateSpace = spaceHierarchy->Lookup(space->getName());
	bool duplicateFound = false;
	if (duplicateSpace != NULL) duplicateFound = true;
	spaceHierarchy->Enter(space->getName(), space, false);
	return !duplicateFound;
}

Space *PartitionHierarchy::getCommonAncestor(Space *space1, Space *space2) {
	if (space1 == space2) return space1;
	if (space1->isParentSpace(space2)) return space2;
	if (space2->isParentSpace(space1)) return space1;
	Space *nextSpace = space1;
	while ((nextSpace = nextSpace->getParent()) != NULL) {
		if (space2->isParentSpace(nextSpace)) return nextSpace;
	}
	return getRootSpace();
}

void PartitionHierarchy::performAllocationAnalysis(int segmentedPPS) {	
	Space *root = getRootSpace();
	List<const char*> *variableList = root->getLocalDataStructureNames();
	
	// instruction to allocate all scalar and non-partitionable variables in the root LPS
	for (int i = 0; i < variableList->NumElements(); i++) {
		DataStructure *variable = root->getStructure(variableList->Nth(i));
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(variable);
		if (array == NULL) {
			variable->setAllocator(root);
			variable->getUsageStat()->flagAllocated();
		}
	}

	// then do breadth first search in the partition hierarchy to make allocation decisions
	std::deque<Space*> lpsQueue;
        lpsQueue.push_back(root);
	while (!lpsQueue.empty()) {
		
		Space *lps = lpsQueue.front();
                lpsQueue.pop_front();

		// setup the segmented PPS property in the LPS here since we are traversing all
		// LPSes and have the information for the setup
		lps->setSegmentedPPS(segmentedPPS);

		List<Space*> *children = lps->getChildrenSpaces();	
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());

		int ppsId = lps->getPpsId();
		
		// iterate over all the arrays of current LPS and consider only those for memory
		// allocation that have been used within any of LPS's compute stages.
		variableList = lps->getLocallyUsedArrayNames();
		for (int i = 0; i < variableList->NumElements(); i++) {
			DataStructure *variable = lps->getLocalStructure(variableList->Nth(i));
			LPSVarUsageStat *stat = variable->getUsageStat();
			if (!(stat->isAccessed() || stat->isReduced())) continue;
			
			// if there are boundary overlappings among the partitions of the array
			// in this LPS then memory need to be allocated for this variable
			ArrayDataStructure *array = (ArrayDataStructure*) variable;
			bool hasOverlapping = array->hasOverlappingsAmongPartitions();
			
			// check if the variable has been allocated before in any ancestor LPS
			DataStructure *lastAllocation = array->getClosestAllocation();

			// if there is no previous allocation for this structure then it should
			// be allocated
			bool notAllocatedBefore = (lastAllocation == NULL);

			// if the structure has been allocated before then a checking should be
			// done to see if the array has been reordered since last allocation. If
			// it has been reordered then again a new allocation is needed.
			bool reordered = false; 
			if (lastAllocation != NULL) {
				Space *allocatingSpace = lastAllocation->getSpace();
				reordered = array->isReorderedAfter(allocatingSpace);
			}

			// even if the array has not been reordered since last allocation, if its 
			// last allocation was above the segmented PPS layer and on a different 
			// PPS than the current LPS has been mapped to then, again, it should be 
			// allocated
			bool lastAllocInaccessible = false;
			if (lastAllocation != NULL) {
				Space *allocatingSpace = lastAllocation->getSpace();
				int lastPpsId = allocatingSpace->getPpsId();
				lastAllocInaccessible = (lastPpsId > segmentedPPS) 
						&& (ppsId != lastPpsId);
			}

			if (hasOverlapping || notAllocatedBefore 
					|| reordered || lastAllocInaccessible) {

				// if the data structure has not been reordered since the last 
				// allocation and the current LPS is a subpatition LPS then we
				// allocate the structure on the parent LPS to keep the number of
				// data parts low
				if (lps->isSubpartitionSpace() && !reordered) {
					Space *parentLps = lps->getParent();
					DataStructure *parentArray = array->getSource();
					parentArray->setAllocator(parentLps);
					parentArray->getUsageStat()->flagAllocated();
				} else {
					array->setAllocator(lps);
					array->getUsageStat()->flagAllocated();
				}
			}
		}	
	}

	// finally do another breadth first search to set up appropriate allocation references for
	// data structures in LPSes that do not allocate them themselves.			
        lpsQueue.push_back(root);
	while (!lpsQueue.empty()) {
		Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
		List<Space*> *children = lps->getChildrenSpaces();	
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		
		variableList = lps->getLocalDataStructureNames();
		for (int i = 0; i < variableList->NumElements(); i++) {
			DataStructure *structure = lps->getLocalStructure(variableList->Nth(i));
			// if the structure is not allocated then try to find a source reference
			// in some ancestor LPS where it has been allocated (this sets up the 
			// forward pointer from a lower to upper LPS)
			if (!structure->getUsageStat()->isAllocated()) {
				DataStructure *lastAllocation = structure->getClosestAllocation();
				if (lastAllocation != NULL) {
					structure->setAllocator(lastAllocation->getSpace());
				}
			// on the other hand, if the structure has been allocated in current LPS
			// then this allocation can be used to set up back references to ancestor
			// LPSes that neither allocate this structure themselves nor have any
			// forward reference to their own ancestor LPSes been set for structure 	
			} else {
				DataStructure *source = structure->getSource();
				while (source != NULL && source->getAllocator() == NULL) {
					source->setAllocator(lps);
					source = source->getSource();
				}
			}
		}
	}
}
