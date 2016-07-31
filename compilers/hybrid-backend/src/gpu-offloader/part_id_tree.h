#ifndef _H_part_id_tree
#define _H_part_id_tree

#include "../utils/list.h"

#include <vector>
#include <iostream>
#include <cstdlib>

/* This header contains the definition of a sorted tree structure for quick lookup and insertion of multidimensional
 * data part IDs. The tree construction process is very much like the part-ID-container tree of the part-tracking
 * library in the memory management module. We did not use that tree for data part tracking during populating GPU
 * stage-in buffers as the earlier tree construction comes up with a lot of additional features and rigidly coupled
 * with storing actual data parts, not just their IDs.
 *
 * We recommend future developer to try to generalize the tree construction in the memory management module by clever
 * use of class inheritence and use that instead of the current class if he/she needs the part-ID-tree construction
 * in other places than just the GPU offloading context. Afterwards, he/she should delete this class and refactore the 
 * LPU part-tracking library that uses it.   
 */

class PartIdNode {
  private:
	std::vector<int> partArray;
	std::vector<PartIdNode*> children;

	// In some cases, we need to know where the part fpr an specific ID is located in a corresponding part holder
	// list if the part ID is found in the tree. This storage index vector maintain this information. Note that
	// is property is only used at the leaf level of the Part-ID-Tree 
	std::vector<int> partStoreIndices;
  public:
	~PartIdNode();
	void print(int indentLevel, std::ostream *stream);
	
	// This function attempts to insert a new part Id in the tree. If the part Id already exists then it returns
	// false; otherwise it returns true. Notice that the argument for the last parameter is by default -1. This
	// is set this way to ignore part-storage-index in cases where it is not relevant.
	bool insertPartId(List<int*> *partId, int partDimensions, int storageIndex = -1);

	// This function just quries an ID in the tree and returns true or false to indicate the ID exists or does
	// not exist respectively.
	bool doesPartExist(List<int*> *partId, int partDimensions);

	// This function returns the saved storage index of the data part associated with an ID that already exists
	// in the current part-ID-tree.
	int getPartStorageIndex(List<int*> *partId, int partDimensions);
  private:
	bool insertPartId(List<int*> *partId, 
			int idLength, int partDimensions, 
			int storageIndex, int currentPos);	
	bool doesPartExist(List<int*> *partId, 
			int idLength, int partDimensions, int currentPos);
	void insertFirstPartId(List<int*> *partId, 
			int idLength, int partDimensions, 
			int storageIndex, int currentPos);
	int getPartStorageIndex(List<int*> *partId, int idLength, int partDimensions, int currentPos);
};

#endif
