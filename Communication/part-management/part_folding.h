#ifndef PART_FOLDING_H_
#define PART_FOLDING_H_

#include "../utils/list.h"
#include "../structure.h"
#include "../utils/interval.h"
#include "part_config.h"
#include <cstdlib>
#include <iostream>
#include <vector>

class FoldStrain;

/* This class represents a folding of part IDs from a part-container for any data structure. The folding can
 * subsequently be used to construct a compact interval set representation of the entire container without. */
class PartFolding {
protected:
	int dimNo;
	int level;
	List<PartFolding*> *descendants;
	Range idRange;
public:
	PartFolding(int id, int dimNo, int level);
	PartFolding(Range *idRange, int dimNo, int level);
	~PartFolding();
	bool isEqualInContent(PartFolding *other);
	void addDescendant(PartFolding *descendant) { descendants->Append(descendant); }
	List<PartFolding*> *getDescendants() { return descendants; }
	Range getIdRange() { return idRange; }
	int getSiblingsCount() { return idRange.max - idRange.min + 1; }
	void print(std::ostream &stream, int indentLevel);

	// recursively removes all descendant part foldings and then deletes the descendants list
	void clearContent();

	// coalesce should only be done if two sub-foldings have the same content
	void coalesce(Range otherIdRange) { this->idRange.max = otherIdRange.max; }

	// generate a list of separate fold paths from the current part folding
	List<FoldStrain*> *extractStrains(std::vector<PartFolding*> *currentFoldingChain
			= new std::vector<PartFolding*>);

	// a function that tries to truncate a folding description at the bottom by eliminating the lower levels if
	// they contain all the parts (i.e., if the range contains 0 to max - 1 part IDs) there are in their respective
	// levels; the first argument is there to restrict such pruning up to a specified level as sometimes there
	// might be a need to keep a fold description expanded up to a minimum number of levels
	void pruneFolding(int lowerLevelBound, DataItemConfig *dataConfig);

	// generates an interval description as a list of multidimensional interval sequences for the entire fold
	List<MultidimensionalIntervalSeq*> *generateIntervalDesc(DataItemConfig *dataConfig);
};

// this class represents a single path in a part-folding; the path progresses bottom up; i.e., initial reference
// holds the tail of the path
class FoldStrain {
private:
	int dimNo;
	int level;
	Range idRange;
	FoldStrain *previous;
public:
	FoldStrain(int dimNo, int level, Range idRange) {
		this->dimNo = dimNo;
		this->level = level;
		this->idRange = idRange;
		this->previous = NULL;
	}
	void setPrevious(FoldStrain *previous) { this->previous = previous; }
	FoldStrain *getPrevious() { return previous; }
	Range getIdRange() { return idRange; }
	int getDimNo() { return dimNo; }
	int getLevel() { return level; }
	void print(std::ostream &stream);

	// creates a new fold strain from the current one that is disconnected from the parent, if exists
	FoldStrain *copy() { return new FoldStrain(dimNo, level, idRange); }

	// generates an interval description as a list of multidimensional interval sequences for the current strain
	List<MultidimensionalIntervalSeq*> *generateIntervalDesc(DataItemConfig *dataConfig);
};

// this class is used to generate separate folding chains per dimensions from a strain of part folding
class DimensionFold {
private:
	int dimNo;
	std::vector<FoldStrain*> *fold;
public:
	DimensionFold(int dimNo, std::vector<FoldStrain*> *fold) {
		this->dimNo = dimNo;
		this->fold = fold;
	}
	~DimensionFold() { delete fold; }
	int getDimNo() { return dimNo; }
	std::vector<FoldStrain*> *getFold() { return fold; }
	void print(std::ostream &stream, int indentLevel);

	// function to be used to get individual dimension chains within a fold strain
	static List<DimensionFold*> *separateDimensionFolds(FoldStrain *foldStrain);

	// generates an interval description as a list of one dimensional interval sequences for the current dimension
	// of some fold strain
	List<IntervalSeq*> *generateIntervalDesc(DataItemConfig *dataConfig);

	// Since the dimensions of sibling parts at any level of the partition hierarchy may be of different lengths,
	// it is not right to just generate interval description for a range of parts by assuming that they have the
	// same dimensionality. The problem has a cumulative effect if we consider a hierarchy of id-range -- that a
	// a dimension fold represents -- and dimension inequality being present in one or more intermediate levels.
	// So the following function should be used before generating interval description for a fold strain to distill
	// it into one or more fold strains that have homogeneous id-ranges at each level except, if applicable, in
	// the lowest level. For the lowest level, the interval description generation process within the partition
	// library takes care of dimension length inequalities.
	List<DimensionFold*> *branchOutForDimHeterogeneity(DataItemConfig *dataConfig);

	// This attempts to reduce the number of levels in the dimension fold to make interval description generation
	// process more efficient. Further, if there are lesser number of levels then the need of branching out to
	// account for dimension length heterogeneity in parts is minimized.
	void pruneFolding(DataItemConfig *dataConfig);

private:
	// a recursive helper function for branchOutForDimHeterogeneity routine's calculation
	List<DimensionFold*> *branchOut(DataItemConfig *dataConfig, std::vector<FoldStrain*> *branchUnderConstruction);
};

#endif
