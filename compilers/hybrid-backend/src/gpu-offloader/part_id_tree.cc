#include "part_id_tree.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <cstdlib>

PartIdNode::~PartIdNode() {
	for (unsigned int i = 0; i < partArray.size(); i++) {
                PartIdNode *child = children[i];
                delete child;
        }
}

void PartIdNode::print(int indentLevel, std::ostream *stream) {
	std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';
        for (unsigned int i = 0; i < partArray.size(); i++) {
                *stream << '\n' << indent.str() << partArray[i] << ':';
                PartIdNode *child = children[i];
                child->print(indentLevel + 1, stream);
        }
}

bool PartIdNode::insertPartId(List<int*> *partId, int partDimensions, int storageIndex) {
	int idLength = partId->NumElements() * partDimensions;
	return insertPartId(partId, idLength, partDimensions, storageIndex, 0);
}

bool PartIdNode::doesPartExist(List<int*> *partId, int partDimensions) {
	int idLength = partDimensions * partId->NumElements();
	return doesPartExist(partId, idLength, partDimensions, 0);
}

int PartIdNode::getPartStorageIndex(List<int*> *partId, int partDimensions) {
	int idLength = partDimensions * partId->NumElements();
	return getPartStorageIndex(partId, idLength, partDimensions, 0);
}

bool PartIdNode::insertPartId(List<int*> *partId, int idLength, 
		int partDimensions, int storageIndex, int currentPos) {
	
	int listIndex = currentPos / partDimensions;
	int dimIndex = currentPos % partDimensions;
	int currPosId = partId->Nth(listIndex)[dimIndex];

	int location = binsearch::locateKey(partArray, currPosId);
	if (location != KEY_NOT_FOUND) {
		if (currentPos == idLength - 1) return false;
		PartIdNode *child = children[location];
		return child->insertPartId(partId, idLength, partDimensions, storageIndex, currentPos + 1);	
	}

	int insertIndex = binsearch::locatePointOfInsert(partArray, currPosId);
	partArray.insert(partArray.begin() + insertIndex, currPosId);
	if (currentPos == idLength - 1) {
		partStoreIndices.insert(partStoreIndices.begin() + insertIndex, storageIndex);
		return true;
	}

	PartIdNode *child = new PartIdNode();
	children.insert(children.begin() + insertIndex, child);
	child->insertFirstPartId(partId, idLength, partDimensions, storageIndex, currentPos + 1);
	return true;
}

bool PartIdNode::doesPartExist(List<int*> *partId, int idLength, int partDimensions, int currentPos) {

	int listIndex = currentPos / partDimensions;
        int dimIndex = currentPos % partDimensions;
        int currPosId = partId->Nth(listIndex)[dimIndex];

	int location = binsearch::locateKey(partArray, currPosId);
	if (location == KEY_NOT_FOUND) return false;

	if (currentPos == idLength - 1) return true;
	
	PartIdNode *child = children[location];
	return child->doesPartExist(partId, idLength, partDimensions, currentPos + 1);
}

void PartIdNode::insertFirstPartId(List<int*> *partId, int idLength, 
		int partDimensions, int storageIndex, int currentPos) {
	
	int listIndex = currentPos / partDimensions;
        int dimIndex = currentPos % partDimensions;
        int currPosId = partId->Nth(listIndex)[dimIndex];
	partArray.push_back(currPosId);

	if (currentPos < idLength - 1) {
		PartIdNode *child = new PartIdNode();
		children.push_back(child);
		child->insertFirstPartId(partId, idLength, partDimensions, storageIndex, currentPos + 1);
	} else {
		partStoreIndices.push_back(storageIndex);
	}
}

int PartIdNode::getPartStorageIndex(List<int*> *partId, int idLength, int partDimensions, int currentPos) {
	int listIndex = currentPos / partDimensions;
        int dimIndex = currentPos % partDimensions;
        int currPosId = partId->Nth(listIndex)[dimIndex];
	int location = binsearch::locateKey(partArray, currPosId);
	if (currentPos == idLength - 1) {
		return partStoreIndices.at(location);
	} else {
		PartIdNode *child = children[location];
		return child->getPartStorageIndex(partId, idLength, partDimensions, currentPos + 1);
	}
}

