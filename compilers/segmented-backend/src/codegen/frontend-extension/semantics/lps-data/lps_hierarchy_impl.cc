#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/partition_function.h"
#include "../../../../../../frontend/src/static-analysis/usage_statistic.h"

#include <string>
#include <sstream>
#include <cstdlib>
#include <stack>
#include <deque>

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
