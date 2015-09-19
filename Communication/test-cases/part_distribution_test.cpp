/* This is test checks if the part-distribution tree that holds data-part configuration for multiple related/independent
 * partitions and multiple segments can be constructed properly.
 * */

#include "../communication/part_distribution.h"
#include "../utils/list.h"
#include "../utils/id_generation.h"
#include <iostream>
#include <vector>

using namespace std;

int mainPDT() {

	BranchingContainer *rootContainer = new BranchingContainer(0, LpsDimConfig());

	// configuration for first partition
	vector<LpsDimConfig> dimOrder1;
	dimOrder1.push_back(LpsDimConfig(0, 0, 1));
	dimOrder1.push_back(LpsDimConfig(0, 1, 1));
	dimOrder1.push_back(LpsDimConfig(1, 0, 2));
	dimOrder1.push_back(LpsDimConfig(1, 1, 2));
	dimOrder1.push_back(LpsDimConfig(2, 1, 3));
	dimOrder1.push_back(LpsDimConfig(2, 0, 3));

	// insert some data part for two segments
	List<int*> *first = idutils::generateIdFromArray(new int[6] {0, 0, 0, 1, 5, 6}, 2, 6);
	rootContainer->insertPart(dimOrder1, 0, first);
	List<int*> *second = idutils::generateIdFromArray(new int[6] {0, 0, 0, 1, 6, 0}, 2, 6);
	rootContainer->insertPart(dimOrder1, 0, second);
	rootContainer->insertPart(dimOrder1, 1, second);
	List<int*> *third = idutils::generateIdFromArray(new int[6] {0, 0, 2, 0, 1, 2}, 2, 6);
	rootContainer->insertPart(dimOrder1, 0, third);
	List<int*> *fourth = idutils::generateIdFromArray(new int[6] {1, 0, 2, 0, 0, 0}, 2, 6);
	rootContainer->insertPart(dimOrder1, 0, fourth);
	rootContainer->insertPart(dimOrder1, 1, fourth);

	// configuration for the second partition
	vector<LpsDimConfig> dimOrder2;
	dimOrder2.push_back(LpsDimConfig(0, 0, 1));
	dimOrder2.push_back(LpsDimConfig(0, 1, 1));
	dimOrder2.push_back(LpsDimConfig(1, 1, 5));
	dimOrder2.push_back(LpsDimConfig(1, 0, 5));

	// insert some data for a new segment
	first = idutils::generateIdFromArray(new int[4] {0, 0, 1, 0}, 2, 4);
	rootContainer->insertPart(dimOrder2, 2, first);
	second = idutils::generateIdFromArray(new int[4] {0, 0, 1, 1}, 2, 4);
	rootContainer->insertPart(dimOrder2, 2, second);

	// configuration for third partition
	vector<LpsDimConfig> dimOrder3;
	dimOrder3.push_back(LpsDimConfig(0, 0, 1));
	dimOrder3.push_back(LpsDimConfig(0, 1, 1));
	dimOrder3.push_back(LpsDimConfig(1, 0, 2));
	dimOrder3.push_back(LpsDimConfig(1, 1, 2));

	// insert some data for first and third segment
	first = idutils::generateIdFromArray(new int[4] {0, 0, 0, 1}, 2, 4);
	rootContainer->insertPart(dimOrder3, 0, first);
	rootContainer->insertPart(dimOrder3, 2, first);
	second = idutils::generateIdFromArray(new int[4] {0, 0, 7, 7}, 2, 4);
	rootContainer->insertPart(dimOrder3, 0, second);

	rootContainer->print(0, cout);

	return 0;
}



