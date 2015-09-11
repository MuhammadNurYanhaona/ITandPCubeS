/* The goal of this test is to verify if we can retrieve part-IDs at any level from a part-tracking-container hierarchy.
 * We need to be able get the super parts from a container hierarchy properly as they may be the confinement roots of
 * communications. Once we get the IDs of the super parts then we will use them to search in the composite multi-branching
 * tree in the communication library to determine the specifics of any particular communication.
 *
 * This test does not do much, it just create a container hierarchy with three levels then ask for the part IDs at different
 * levels for visual inspection.
 * */

#include "../part-management/part_tracking.h"
#include "../utils/list.h"
#include "../utils/id_generation.h"
#include <cstdlib>
#include <iostream>

using namespace std;

int mainPIRT() {

	vector<DimConfig> dimOrder;
	dimOrder.push_back(DimConfig(0, 0));
	dimOrder.push_back(DimConfig(0, 1));
	dimOrder.push_back(DimConfig(1, 0));
	dimOrder.push_back(DimConfig(1, 1));
	dimOrder.push_back(DimConfig(2, 1));
	dimOrder.push_back(DimConfig(2, 0));

	PartIdContainer *partIdContainer = new PartListContainer(dimOrder[0]);
	List<int*> *first = generateIdFromArray(new int[6] {0, 0, 0, 1, 5, 6}, 2, 6);
	partIdContainer->insertPartId(first, 2, dimOrder);
	List<int*> *second = generateIdFromArray(new int[6] {0, 0, 0, 1, 6, 0}, 2, 6);
	partIdContainer->insertPartId(second, 2, dimOrder);
	List<int*> *third = generateIdFromArray(new int[6] {0, 0, 2, 0, 1, 2}, 2, 6);
	partIdContainer->insertPartId(third, 2, dimOrder);
	List<int*> *fourth = generateIdFromArray(new int[6] {1, 0, 2, 0, 0, 0}, 2, 6);
	partIdContainer->insertPartId(fourth, 2, dimOrder);

	cout << "Part Hierarchy\n";
	partIdContainer->print(0, cout);

	cout << "\nLevel 2 IDs: \n";
	List<List<int*>*> *level2PartIdList = partIdContainer->getAllPartIdsAtLevel(2, 2);
	for (int i = 0; i < level2PartIdList->NumElements(); i++) {
		List<int*> *partId = level2PartIdList->Nth(i);
		for (int j = 0; j < partId->NumElements(); j++) {
			int *id = partId->Nth(j);
			cout << "[" << id[0] << ", " << id[1] << "]";
		}
		cout << "\n";
	}

	cout << "\nLevel 1 IDs: \n";
	List<List<int*>*> *level1PartIdList = partIdContainer->getAllPartIdsAtLevel(1, 2);
	for (int i = 0; i < level1PartIdList->NumElements(); i++) {
		List<int*> *partId = level1PartIdList->Nth(i);
		for (int j = 0; j < partId->NumElements(); j++) {
			int *id = partId->Nth(j);
			cout << "[" << id[0] << ", " << id[1] << "]";
		}
		cout << "\n";
	}

	cout << "\nLevel 0 IDs: \n";
	List<List<int*>*> *level0PartIdList = partIdContainer->getAllPartIdsAtLevel(0, 2);
	for (int i = 0; i < level0PartIdList->NumElements(); i++) {
		List<int*> *partId = level0PartIdList->Nth(i);
		for (int j = 0; j < partId->NumElements(); j++) {
			int *id = partId->Nth(j);
			cout << "[" << id[0] << ", " << id[1] << "]";
		}
		cout << "\n";
	}

	return 0;
}



