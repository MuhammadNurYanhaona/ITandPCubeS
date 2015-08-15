/* The goal of this test is to check if fold strains (a strain is a path within a parts folding) are
 * generated correctly from multidimensional folds. In addition, it checks if our process of extracting
 * sub-strains from a multidimensional strain is accurate.
 * */

#include "../part-management/part_folding.h"
#include "../part-management/part_tracking.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

int main() {
	vector<DimConfig> dimOrder;
	dimOrder.push_back(DimConfig(0, 0));
	dimOrder.push_back(DimConfig(0, 1));
	dimOrder.push_back(DimConfig(1, 1));
	dimOrder.push_back(DimConfig(1, 0));
	dimOrder.push_back(DimConfig(1, 2));

	PartIdContainer *partContainer = new PartListContainer(dimOrder[0]);
	List<int*> *entry1 = new List<int*>;
	int e11[3] = {0, 0, 0}; int e12[3] = {0, 0, 0};
	entry1->Append(e11);entry1->Append(e12);
	partContainer->insertPartId(entry1, 3, dimOrder);
	List<int*> *entry2 = new List<int*>;
	int e21[3] = {0, 1, 0}; int e22[3] = {1, 0, 0};
	entry2->Append(e21);entry2->Append(e22);
	partContainer->insertPartId(entry2, 3, dimOrder);
	List<int*> *entry3 = new List<int*>;
	int e31[3] = {0, 1, 0}; int e32[3] = {1, 0, 1};
	entry3->Append(e31);entry3->Append(e32);
	partContainer->insertPartId(entry3, 3, dimOrder);
	List<int*> *entry4 = new List<int*>;
	int e41[3] = {1, 1, 0}; int e42[3] = {1, 1, 1};
	entry4->Append(e41);entry4->Append(e42);
	partContainer->insertPartId(entry4, 3, dimOrder);
	List<int*> *entry5 = new List<int*>;
	int e51[3] = {1, 1, 0}; int e52[3] = {1, 1, 2};
	entry5->Append(e51);entry5->Append(e52);
	partContainer->insertPartId(entry5, 3, dimOrder);
	List<int*> *entry6 = new List<int*>;
	int e61[3] = {1, 1, 0}; int e62[3] = {2, 1, 1};
	entry6->Append(e61);entry6->Append(e62);
	partContainer->insertPartId(entry6, 3, dimOrder);
	List<int*> *entry7 = new List<int*>;
	int e71[3] = {1, 1, 0}; int e72[3] = {2, 1, 2};
	entry7->Append(e71);entry7->Append(e72);
	partContainer->insertPartId(entry7, 3, dimOrder);

	cout << "Container Description:-----------------------------------";
	partContainer->print(0, cout);
	List<PartFolding*> *fold = new List<PartFolding*>;
	partContainer->foldContainer(fold);
	cout << "\n\nFold description:------------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		fold->Nth(i)->print(cout, 0);
	}
	cout << "\n\nFold strains:----------------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		PartFolding *folding = fold->Nth(i);
		List<FoldStrain*> *foldStrains = folding->extractStrains();
		for (int j = 0; j < foldStrains->NumElements(); j++) {
			foldStrains->Nth(j)->print(cout);
		}
	}
	cout << "\n\nDimension Folds:--------------------------------------";
	for (int i = 0; i < fold->NumElements(); i++) {
		PartFolding *folding = fold->Nth(i);
		List<FoldStrain*> *foldStrains = folding->extractStrains();
		for (int j = 0; j < foldStrains->NumElements(); j++) {
			FoldStrain* strain = foldStrains->Nth(j);
			List<DimensionFold*> *dimFolds = DimensionFold::separateDimensionFolds(strain);
			cout << '\n';
			for (int k = 0; k < dimFolds->NumElements(); k++) {
				dimFolds->Nth(k)->print(cout);
			}
		}
	}

	return 0;
}


