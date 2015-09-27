/* This test generates a part-container hierarchy, folds the hierarchy into a compact description, tries to prune it up
 * to a desired level, then describes the fold as list of, possibly multidimensional, interval sequences.
 * */

#include "../structure.h"
#include "../part-management/part_folding.h"
#include "../part-management/part_tracking.h"
#include "../part-management/part_config.h"
#include "../utils/id_generation.h"
#include "../utils/list.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

int mainFPT() {

	// partitioned dimensions' priority orders
	vector<DimConfig> dimOrder;
	dimOrder.push_back(DimConfig(0, 0));
	dimOrder.push_back(DimConfig(0, 1));
	dimOrder.push_back(DimConfig(1, 1));
	dimOrder.push_back(DimConfig(1, 0));
	dimOrder.push_back(DimConfig(2, 0));
	dimOrder.push_back(DimConfig(2, 1));

	// data partition configuration
	DataItemConfig *dataConfig = new DataItemConfig(2, 3);
	dataConfig->setDimension(0, Dimension(100));
	dataConfig->setDimension(1, Dimension(100));
	dataConfig->setPartitionInstr(0, 0, new BlockSizeInstr(Dimension(100), 0, 20));
	dataConfig->setPartitionInstr(0, 1, new BlockSizeInstr(Dimension(100), 0, 20));
	dataConfig->setPartitionInstr(1, 0, new BlockCountInstr(Dimension(100), 0, 2));
	dataConfig->setPartitionInstr(1, 1, new BlockCountInstr(Dimension(100), 0, 2));
	dataConfig->setPartitionInstr(2, 0, new StrideInstr(Dimension(100), 0, 2));
	dataConfig->setPartitionInstr(2, 1, new StrideInstr(Dimension(100), 0, 2));
	dataConfig->updateParentLinksOnPartitionConfigs();

	// create a part container hierarchy
	List<int*> *partId = NULL;
	PartIdContainer *rootContainer = new PartListContainer(dimOrder[0]);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 0, 0, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 0, 0, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 1, 0, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 1, 0, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 0, 0, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 0, 0, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 1, 0, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 0, 1, 0, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 0, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 0, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 0, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 0, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 1, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 1, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 1, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 1, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 0, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 0, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 0, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 0, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 1, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 0, 1, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 1, 1, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {0, 1, 1, 1, 1, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {4, 4, 0, 1, 0, 0}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);
	partId = idutils::generateIdFromArray(new int[6] {4, 4, 0, 1, 0, 1}, 2, 6);
	rootContainer->insertPartId(partId, 2, dimOrder);

	cout << "container hierarchy:---------------------------------------";
	rootContainer->print(0, cout);

	List<PartFolding*> *fold = new List<PartFolding*>;
	rootContainer->foldContainer(fold);
	cout << "\n\nFold description:--------------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		fold->Nth(i)->print(cout, 0);
	}

	cout << "\n\nPruned Fold description:-------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		fold->Nth(i)->pruneFolding(0, dataConfig);
		fold->Nth(i)->print(cout, 0);
	}

	cout << "\n\nFold strains:------------------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		PartFolding *folding = fold->Nth(i);
		List<FoldStrain*> *foldStrains = folding->extractStrains();
		for (int j = 0; j < foldStrains->NumElements(); j++) {
			FoldStrain* foldStrain = foldStrains->Nth(j);
			foldStrain->print(cout);
			List<DimensionFold*> *dimensionFolds = DimensionFold::separateDimensionFolds(foldStrain);
			cout << "\n\tDimension folds:\n";
			for (int k = 0; k < dimensionFolds->NumElements(); k++) {
				dimensionFolds->Nth(k)->print(cout, 1);
			}
		}
	}

	cout << "\n\nInterval Sequences for Folds:--------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		PartFolding *folding = fold->Nth(i);
		cout << "\nFold #" << i + 1 << ":---------------------------------------\n";
		List<MultidimensionalIntervalSeq*> *intList = folding->generateIntervalDesc(dataConfig);
		for (int j = 0; j < intList->NumElements(); j++) {
			intList->Nth(j)->draw();
		}
	}

	return 0;
}
