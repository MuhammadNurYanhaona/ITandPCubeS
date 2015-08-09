/* This test case creates a part container having four levels
 * then inserts into it randomly generated parts
 * then checks if the parts are inserted correctly by validating unique parts count
 * then checks if part searching process is performing as expected by first searching
 * inserted parts in random order and afterwards in the order they are stored within the container
 * */

#include "../part_tracking.h"
#include <cstdlib>
#include <iostream>

using namespace std;

int mainPCT() {

	vector<DimConfig> dimOrder;
	dimOrder.push_back(DimConfig(0, 0));
	dimOrder.push_back(DimConfig(0, 1));
	dimOrder.push_back(DimConfig(1, 0));
	dimOrder.push_back(DimConfig(1, 1));

	PartIdContainer *partIdContainer = NULL;
	if (dimOrder.size() > 1) partIdContainer = new PartListContainer(dimOrder[0]);
	else partIdContainer = new PartContainer(dimOrder[0]);

	vector<int> array;
	srand(time(NULL));

	List<List<int*>*> *partStorage = new List<List<int*>*>;
	int uniquePartsCount = 0;
	for (int i = 0; i < 5; i++) {
		int i0 = rand() % 10;
		for (int l = 0; l < 2; l++) {
			int i1 = rand() % 2;
			for (int j = 0; j < 5; j++) {
				int j0 = rand() % 10;
				for (int k = 0; k < 5; k++) {
					int j1 = rand() % 5;
					List<int*> *partId = new List<int*>;
					partId->Append(new int[2]);
					partId->Nth(0)[0] = i0;
					partId->Nth(0)[1] = i1;
					partId->Append(new int[2]);
					partId->Nth(1)[0] = j0;
					partId->Nth(1)[1] = j1;
					partStorage->Append(partId);
					bool status = partIdContainer->insertPartId(partId, 2, dimOrder, 0);
					if (status) uniquePartsCount++;
				}
			}
		}
	}

	partIdContainer->postProcess();
	partIdContainer->print(0, std::cout);
	cout.flush();


	int partsFound = 0;
	PartIterator *iterator = partIdContainer->getIterator();
	SuperPart *part = NULL;
	List<List<int*>*> *partStorage2 = new List<List<int*>*>;
	while ((part = iterator->getCurrentPart()) != NULL) {
		partStorage2->Append(part->getPartId());
		iterator->advance();
		partsFound++;
	}

	cout << "\n\nIterator validation:";
	cout << "\nParts inserted: " << uniquePartsCount << " Parts found: " << partsFound;

	cout << "\n\nSearch Validation #1: random part order search\n";
	for (int i = 0; i < partStorage->NumElements(); i++) {
		partIdContainer->getPart(partStorage->Nth(i), iterator, 2);
	}
	iterator->printStats(cout, 0);

	cout << "\n\nSearch Validation #2: sorted part order search\n";
	iterator->resetStats();
	for (int i = 0; i < partStorage2->NumElements(); i++) {
		partIdContainer->getPart(partStorage2->Nth(i), iterator, 2);
	}
	iterator->printStats(cout, 0);

	return 0;
}


