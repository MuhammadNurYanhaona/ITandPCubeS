#include "part_folding.h"
#include "../utils/list.h"
#include "../structure.h"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

//------------------------------------------------------------- Part Folding ------------------------------------------------------------/

PartFolding::PartFolding(int id, int dimNo, int level) {
	this->dimNo = dimNo;
	this->level = level;
	this->idRange.min = id;
	this->idRange.max = id;
	descendants = new List<PartFolding*>;
}

PartFolding::PartFolding(Range *idRange, int dimNo, int level) {
	this->dimNo = dimNo;
	this->level = level;
	this->idRange.min = idRange->min;
	this->idRange.max = idRange->max;
	descendants = new List<PartFolding*>;
}

PartFolding::~PartFolding() {
	while (descendants->NumElements() > 0) {
		PartFolding *child = descendants->Nth(0);
		descendants->RemoveAt(0);
		delete child;
	}
	delete descendants;
}

bool PartFolding::isEqualInContent(PartFolding *other) {
	if (descendants == NULL && other->descendants == NULL) return true;
	if ((descendants != NULL && other->descendants == NULL)
			|| (descendants == NULL && other->descendants != NULL)) return false;
	if (descendants->NumElements() != other->descendants->NumElements()) return false;
	for (int i = 0; i < descendants->NumElements(); i++) {
		PartFolding *myChild = descendants->Nth(i);
		PartFolding *othersChild = other->descendants->Nth(i);
		if (!myChild->getIdRange().isEqual(othersChild->getIdRange())) return false;
		if (!myChild->isEqualInContent(othersChild)) return false;
	}
	return true;
}

void PartFolding::print(std::ostream &stream, int indentLevel) {
	stream << '\n';
	for (int i = 0; i < indentLevel; i++) stream << '\t';
	stream << idRange.min << "-" << idRange.max;
	for (int i = 0; i < descendants->NumElements(); i++) {
		descendants->Nth(i)->print(stream, indentLevel + 1);
	}
}

List<FoldStrain*> *PartFolding::extractStrains(std::vector<PartFolding*> *currentFoldingChain) {
	List<FoldStrain*> *strains = new List<FoldStrain*>;
	if (descendants->NumElements() != 0) {
		currentFoldingChain->push_back(this);
		for (int i = 0; i < descendants->NumElements(); i++) {
			PartFolding *descendant = descendants->Nth(i);
			List<FoldStrain*> *descStrains = descendant->extractStrains(currentFoldingChain);
			strains->AppendAll(descStrains);
			delete descStrains;
		}
		currentFoldingChain->pop_back();
	} else {
		FoldStrain *foldStrain = new FoldStrain(dimNo, level, idRange);
		FoldStrain *currentStrain = foldStrain;
		for (int i = currentFoldingChain->size() - 1; i >= 0; i--) {
			PartFolding *ancestor = (*currentFoldingChain)[i];
			FoldStrain *prevStrain = new FoldStrain(ancestor->dimNo, ancestor->level, ancestor->idRange);
			currentStrain->setPrevious(prevStrain);
			currentStrain = prevStrain;
		}
		strains->Append(foldStrain);
	}
	return strains;
}

//-------------------------------------------------------------- Fold Strain ------------------------------------------------------------/

void FoldStrain::print(std::ostream &stream) {
	if (previous != NULL) previous->print(stream);
	if (previous == NULL) stream << "\n";
	stream << "[dim: " << dimNo << ", range: " << idRange.min << "-" << idRange.max << "]";
}

//------------------------------------------------------------ Dimension Fold -----------------------------------------------------------/

List<DimensionFold*> *DimensionFold::separateDimensionFolds(FoldStrain *foldStrain) {
	List<int> *foundDimensionList = new List<int>;
	List<vector<FoldStrain*>*> *dimFoldsInConstruct = new List<vector<FoldStrain*>*>;
	FoldStrain *currentFold = foldStrain;
	while (currentFold != NULL) {
		int dimensionNo = currentFold->getDimNo();
		vector<FoldStrain*> *dimVector = NULL;
		for (int i = 0; i < foundDimensionList->NumElements(); i++) {
			if (foundDimensionList->Nth(i) == dimensionNo) {
				dimVector = dimFoldsInConstruct->Nth(i);
				break;
			}
		}
		if (dimVector == NULL) {
			dimVector = new vector<FoldStrain*>;
			dimFoldsInConstruct->Append(dimVector);
			foundDimensionList->Append(dimensionNo);
		}
		FoldStrain *copy = currentFold->copy();
		dimVector->insert(dimVector->begin(), copy);
		currentFold = currentFold->getPrevious();
	}
	List<DimensionFold*> *dimensionFoldList = new List<DimensionFold*>;
	for (int i = 0; i < foundDimensionList->NumElements(); i++) {
		int dimension = foundDimensionList->Nth(i);
		vector<FoldStrain*> *foldDesc = dimFoldsInConstruct->Nth(i);
		DimensionFold *dimensionFold = new DimensionFold(dimension, foldDesc);
		dimensionFoldList->Append(dimensionFold);
	}
	delete foundDimensionList;
	delete dimFoldsInConstruct;
	return dimensionFoldList;
}

void DimensionFold::print(std::ostream &stream) {
	stream << '[' << dimNo << ": ";
	for (int i = 0; i < fold->size(); i++) {
		stream << '(';
		FoldStrain *strain = fold->at(i);
		stream << strain->getIdRange().min;
		stream << "-";
		stream << strain->getIdRange().max;
		stream << ')';
	}
	stream << ']';
}

