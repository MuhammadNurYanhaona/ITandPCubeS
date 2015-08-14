/* This test case creates a part container
 * then inserts a series of randomly generated parts in the container
 * then displays the container contents
 * next it folds the container and displays the fold description.
 * You are supposed to match the fold with the content display to verify the folding is appropriate.
 *
 * Change the three static constants' values to vary the structure of the part container
 **/

#include "../part-management/part_folding.h"
#include "../part-management/part_tracking.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

// determines the number of levels in the container
static const int NUM_OF_LEVELS = 4;
// specifies how full will be each level of the container expressed in the scale of 0 to 1
// the number of entries in this array should be equal to the number of levels
static const float LEVEL_DENSITY[] = {0.2f, 0.9f, 1.0f, 1.0f};
// specifies how many elements should be there at each level of the container
static const int ELEM_PER_LEVEL = 5;

bool contains(List<int> *list, int number) {
	for (int i = 0; i < list->NumElements(); i++) {
		if (list->Nth(i) == number) return true;
	}
	return false;
}

void insertParts(PartIdContainer *rootContainer, vector<DimConfig> dimOrder,
		int currentLevel, List<int*> *partId) {
	int elementCount = ceil(ELEM_PER_LEVEL / LEVEL_DENSITY[currentLevel]);
	List<int> *alreadyEntered = new List<int>;
	while (alreadyEntered->NumElements() < ELEM_PER_LEVEL) {
		int nextInt = rand() % elementCount;
		if (contains(alreadyEntered, nextInt)) continue;
		alreadyEntered->Append(nextInt);
		int *nextId = new int;
		*nextId = nextInt;
		List<int*> *newPartId = new List<int*>;
		newPartId->AppendAll(partId);
		newPartId->Append(nextId);
		if (currentLevel == NUM_OF_LEVELS - 1) {
			rootContainer->insertPartId(newPartId, 1, dimOrder, 0);
		} else {
			insertParts(rootContainer, dimOrder, currentLevel + 1, newPartId);
		}
	}
}

int mainCFT() {

	vector<DimConfig> dimOrder;
	for (int i = 0; i < NUM_OF_LEVELS; i++) {
		dimOrder.push_back(DimConfig(i, 0));
	}

	PartIdContainer *partIdContainer = NULL;
	if (dimOrder.size() > 1) partIdContainer = new PartListContainer(dimOrder[0]);
	else partIdContainer = new PartContainer(dimOrder[0]);

	srand(time(NULL));
	insertParts(partIdContainer, dimOrder, 0, new List<int*>);
	cout << "container description:-----------------------------------";
	partIdContainer->print(0, cout);

	List<PartFolding*> *fold = new List<PartFolding*>;
	partIdContainer->foldContainer(fold);
	cout << "\n\nFold description:-----------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		fold->Nth(i)->print(cout, 0);
	}

	return 0;
}
