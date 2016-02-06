#include "part_folding.h"
#include "../utils/list.h"
#include "../runtime/structure.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <deque>

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
	clearContent();
}

void PartFolding::clearContent() {
	if (descendants != NULL) {
		while (descendants->NumElements() > 0) {
			PartFolding *child = descendants->Nth(0);
			descendants->RemoveAt(0);
			delete child;
		}
		delete descendants;
	}
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
	if (descendants != NULL) {
		for (int i = 0; i < descendants->NumElements(); i++) {
			descendants->Nth(i)->print(stream, indentLevel + 1);
		}
	}
}

List<FoldStrain*> *PartFolding::extractStrains(std::vector<PartFolding*> *currentFoldingChain) {

	List<FoldStrain*> *strains = new List<FoldStrain*>;
	if (descendants != NULL && descendants->NumElements() != 0) {
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

			// do not include the placeholder root level of the part folding in the generated strain, if exists
			if (ancestor->level == -1) break;

			FoldStrain *prevStrain = new FoldStrain(ancestor->dimNo, ancestor->level, ancestor->idRange);
			currentStrain->setPrevious(prevStrain);
			currentStrain = prevStrain;
		}
		strains->Append(foldStrain);
	}
	return strains;
}

void PartFolding::pruneFolding(int lowerLevelBound, DataItemConfig *dataConfig) {

	// skipping the placeholder root level of the fold from setup configuration (the placeholder may not exists)
	if (level == -1) {
		for (int i = 0; i < descendants->NumElements(); i++) {
			PartFolding *nextFold = descendants->Nth(i);
			nextFold->pruneFolding(lowerLevelBound, dataConfig);
		}
	} else {
		dataConfig->adjustDimensionAndPartsCountAtLevel(level, dimNo);
		dataConfig->setPartIdAtLevel(level, dimNo, idRange.min);

		// condition for intermediate folding state
		if (descendants != NULL && descendants->NumElements() != 0) {

			// determine the number of sub-partitions possible in the immediate next level
			PartFolding *firstChild = descendants->Nth(0);
			int nextLevel = firstChild->getLevel();
			int nextDimension = firstChild->getDimNo();
			dataConfig->adjustDimensionAndPartsCountAtLevel(nextLevel, nextDimension);
			int partsCount = dataConfig->getInstruction(nextLevel, nextDimension)->getPartsCount();

			// to eliminate the nested level from the folding configuration, individual descendants themselves
			// should be pruned and the number of descendants the current folding level holds should be equal to
			// the number of sub-partitions
			int prunedDescendants = 0;
			int descendantsCount = descendants->NumElements();
			for (int i = 0; i < descendantsCount; i++) {
				PartFolding *nextFold = descendants->Nth(i);
				nextFold->pruneFolding(lowerLevelBound, dataConfig);
				if (nextFold->descendants == NULL) {
					prunedDescendants += nextFold->getSiblingsCount();
				}
			}

			// prune the lower level only if the next level lies below the pruning level bound
			if (nextLevel > lowerLevelBound && prunedDescendants == partsCount) {
				clearContent();
				descendants = NULL;
			}

		// condition for leaf level folding state; this sets the descendants list to NULL to indicate the default
		// prune-ability of the lowest level
		} else {
			clearContent();
			descendants = NULL;
		}
	}
}

List<MultidimensionalIntervalSeq*> *PartFolding::generateIntervalDesc(DataItemConfig *dataConfig) {
	List<MultidimensionalIntervalSeq*> *dataDesc = new List<MultidimensionalIntervalSeq*>;
	List<FoldStrain*> *foldStrainList = extractStrains();
	for (int i = 0; i < foldStrainList->NumElements(); i++) {
		FoldStrain *strain = foldStrainList->Nth(i);
		List<MultidimensionalIntervalSeq*> *strainSequences = strain->generateIntervalDesc(dataConfig);
		dataDesc->AppendAll(strainSequences);
	}
	while (foldStrainList->NumElements() > 0) {
		FoldStrain *strain = foldStrainList->Nth(0);
		foldStrainList->RemoveAt(0);
		delete strain;
	}
	return dataDesc;
}

//-------------------------------------------------------------- Fold Strain ------------------------------------------------------------/

void FoldStrain::print(std::ostream &stream) {
	if (previous != NULL) previous->print(stream);
	if (previous == NULL) stream << "\n";
	stream << "[dim: " << dimNo << ", range: " << idRange.min << "-" << idRange.max << "]";
}

List<MultidimensionalIntervalSeq*> *FoldStrain::generateIntervalDesc(DataItemConfig *dataConfig) {

	List<DimensionFold*> *dimensionFoldList = DimensionFold::separateDimensionFolds(this);
	List<List<IntervalSeq*>*> *dimensionalIntervalDesc = new List<List<IntervalSeq*>*>;
	int dimensions = dataConfig->getDimensionality();

	// dimension folds are not ordered from the lowest to the highest dimension; therefore we need to get them into
	// proper order before proceeding to combine their interval descriptions
	for (int d = 0; d < dimensions; d++) {
		for (int i = 0; i < dimensionFoldList->NumElements(); i++) {
			DimensionFold *dimensionFold = dimensionFoldList->Nth(i);
			int dimensioNo = dimensionFold->getDimNo();
			if (d == dimensioNo) {
				// we make an attempt to prune the dimension fold into a maximally compact fold-strain path to make
				// the final interval description for it also compact
				dimensionFold->pruneFolding(dataConfig);

				// Note that during folding part containers attention has not been given to ensure there is no miss
				// match in the dimension lengths in the interim layers of a fold. In fact, that was a difficult
				// logic to integrate into folding. But presence of heterogeneity in a fold may lead to erroneous
				// interval description generation. So we break a dimension fold into a set of homogeneous branches.
				List<DimensionFold*> *homogeneousBrances = dimensionFold->branchOutForDimHeterogeneity(dataConfig);
				while (homogeneousBrances->NumElements() > 0) {
					DimensionFold *branch = homogeneousBrances->Nth(0);
					List<IntervalSeq*> *dimensionalSeqs = branch->generateIntervalDesc(dataConfig);
					dimensionalIntervalDesc->Append(dimensionalSeqs);
					homogeneousBrances->RemoveAt(0);
					delete branch;
				}
				break;
			}
		}
	}
	List<MultidimensionalIntervalSeq*> *intervalList = MultidimensionalIntervalSeq::generateIntervalSeqs(
			dimensions, dimensionalIntervalDesc);

	// delete the lists holding 1D interval sequences but not the sequences themselves as they will remain as parts
	// of the multidimensional interval sequences
	while (dimensionalIntervalDesc->NumElements() > 0) {
		List<IntervalSeq*> *oneDSeqList = dimensionalIntervalDesc->Nth(0);
		dimensionalIntervalDesc->RemoveAt(0);
		delete oneDSeqList;
	}
	delete dimensionalIntervalDesc;

	// delete dimension folds
	while (dimensionFoldList->NumElements() > 0) {
		DimensionFold *dimensionFold = dimensionFoldList->Nth(0);
		dimensionFoldList->RemoveAt(0);
		delete dimensionFold;
	}
	delete dimensionFoldList;

	return intervalList;
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

void DimensionFold::print(std::ostream &stream, int indentLevel) {
	for (int i = 0; i < indentLevel; i++) {
		stream << '\t';
	}
	stream << "Dimension " << dimNo << " [";
	for (unsigned int i = 0; i < fold->size(); i++) {
		if (i > 0) stream << ", ";
		FoldStrain *strain = fold->at(i);
		stream << "level " << strain->getLevel() << ": ";
		stream << strain->getIdRange().min;
		stream << "-";
		stream << strain->getIdRange().max;
	}
	stream << ']' << "\n";
}

List<IntervalSeq*> *DimensionFold::generateIntervalDesc(DataItemConfig *dataConfig) {
	int foldLength = fold->size();
	List<Range> *idRangeList = new List<Range>;
	for (int i = 0; i < foldLength; i++) {
		FoldStrain *strain = fold->at(i);
		int levelNo = strain->getLevel();
		Range range = strain->getIdRange();
		idRangeList->Append(range);
		// set the parent dimension information that the partition instruction at current level divides and the parts count
		dataConfig->adjustDimensionAndPartsCountAtLevel(levelNo, dimNo);
		// this is probably not needed for correct interval description generation
		dataConfig->setPartIdAtLevel(levelNo, dimNo, range.min);
	}
	int lastLevel = fold->at(foldLength - 1)->getLevel();
	PartitionInstr *instr = dataConfig->getInstruction(lastLevel, dimNo);
	List<IntervalSeq*> *intervalSeqList = new List<IntervalSeq*>;
	instr->getIntervalDescForRangeHierarchy(idRangeList, intervalSeqList);
	delete idRangeList;
	return intervalSeqList;
}

List<DimensionFold*> *DimensionFold::branchOutForDimHeterogeneity(DataItemConfig *dataConfig) {
	vector<FoldStrain*> *constrVector = new vector<FoldStrain*>;
	List<DimensionFold*> *foldList = branchOut(dataConfig, constrVector);
	delete constrVector;
	return foldList;
}

void DimensionFold::pruneFolding(DataItemConfig *dataConfig) {

	// there must be at least one level in the fold to do any computation over it
	int foldLength = fold->size();
	if (foldLength > 1) {
		for (int i = 0; i < foldLength; i++) {
			FoldStrain *strain = fold->at(i);
			int levelNo = strain->getLevel();
			Range range = strain->getIdRange();
			// set the parent dimension information that the partition instruction at current level divides and the parts count
			dataConfig->adjustDimensionAndPartsCountAtLevel(levelNo, dimNo);
			// this is probably not needed for correct interval description generation
			dataConfig->setPartIdAtLevel(levelNo, dimNo, range.min);
		}

		// to take an upper level into consideration for pruning bottom-most levels should have all the parts
		int pruningLevel = foldLength;
		for (int i = foldLength - 1; i > 0; i--) {
			FoldStrain *strain = fold->at(i);
			Range range = strain->getIdRange();
			PartitionInstr *instr = dataConfig->getInstruction(strain->getLevel(), dimNo);
			int count = instr->getPartsCount();
			if (count == range.max - range.min + 1) {
				pruningLevel = i;
			} else break;
		}

		// prune levels that are full
		while (fold->size() > pruningLevel) {
			fold->pop_back();
		}
	}
}

List<DimensionFold*> *DimensionFold::branchOut(DataItemConfig *dataConfig, vector<FoldStrain*> *constructionBranch) {

	List<DimensionFold*> *dimensionFoldList = new List<DimensionFold*>;
	int updatePoint = constructionBranch->size();
	int foldLength = fold->size();
	FoldStrain *foldStrain = fold->at(updatePoint);
	Range idRange = foldStrain->getIdRange();

	// the last step of a fold needs not be broken into parts as the interval description generation scheme for id-ranges
	// take into consideration any differences in the shape of the interval sequence for individual entries
	if (updatePoint < foldLength - 1) {
		// first we need to set up the dimension length and parts count info correctly in the current level; this will be
		// needed for correct lower level dimension calculation
		dataConfig->adjustDimensionAndPartsCountAtLevel(updatePoint, dimNo);
		// retrieve the partition function used to break the dimension at this level
		PartitionInstr *instr = dataConfig->getInstruction(updatePoint, dimNo);
		instr->setPartId(idRange.min);
		int minId = idRange.min;
		int lastDimensionLength = instr->getDimension().length;
		// go over the individual parts and see if the dimension lengths are the same
		for (int i = minId + 1; i <= idRange.max; i++) {
			instr->setPartId(i);
			int currDimensionLength = instr->getDimension().length;
			// if there is a change in the dimension length at the id then all previously encountered ids since last sub-
			// group formation should constitute a new sub-group
			if (currDimensionLength != lastDimensionLength) {
				Range subRange = Range(minId, i - 1);
				// create a new fold-strain for the subgroup
				FoldStrain *newStrain = new FoldStrain(dimNo, updatePoint, subRange);

				// let the recursive branching process proceed for the newly identified sub-group
				constructionBranch->push_back(newStrain);
				instr->setPartId(minId);
				List<DimensionFold*> *branches = branchOut(dataConfig, constructionBranch);

				// add all the branches returned from lower levels for the current sub-group
				dimensionFoldList->AppendAll(branches);
				delete branches;

				// remove the current sub-group from the construction vector to move to the calculation for the next
				constructionBranch->pop_back();

				// set up the starting point of next sub-group and the dimension length each entry in it should match
				minId = i;
				lastDimensionLength = currDimensionLength;
			}
		}
		// process the last sub-group that is left by the logic of the loop above
		Range subRange = Range(minId, idRange.max);
		FoldStrain *newStrain = new FoldStrain(dimNo, updatePoint, subRange);
		constructionBranch->push_back(newStrain);
		instr->setPartId(minId);
		List<DimensionFold*> *branches = branchOut(dataConfig, constructionBranch);
		dimensionFoldList->AppendAll(branches);
		delete branches;
		constructionBranch->pop_back();
	} else {
		// for the last strain; just append it in the construction vector, create a new dimension fold, and return it
		vector<FoldStrain*> *branchVector = new vector<FoldStrain*>;
		branchVector->reserve(foldLength);
		branchVector->insert(branchVector->begin(), constructionBranch->begin(), constructionBranch->end());
		branchVector->push_back(foldStrain);
		DimensionFold *branch = new DimensionFold(dimNo, branchVector);
		dimensionFoldList->Append(branch);
	}
	return dimensionFoldList;
}

