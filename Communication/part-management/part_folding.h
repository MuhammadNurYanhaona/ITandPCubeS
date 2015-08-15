#ifndef PART_FOLDING_H_
#define PART_FOLDING_H_

#include "../utils/list.h"
#include "../structure.h"
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
	// coalesce should only be done if two sub-foldings have the same content
	void coalesce(Range otherIdRange) { this->idRange.max = otherIdRange.max; }
	void addDescendant(PartFolding *descendant) { descendants->Append(descendant); }
	List<PartFolding*> *getDescendants() { return descendants; }
	Range getIdRange() { return idRange; }
	void print(std::ostream &stream, int indentLevel);
	// generate a list of separate fold paths from the current part folding
	List<FoldStrain*> *extractStrains(std::vector<PartFolding*> *currentFoldingChain = new std::vector<PartFolding*>);
};

// this class represents a single path in a part-folding
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
	~DimensionFold() {
		int size = 0;
		while ((size = fold->size()) > 0) {
			FoldStrain *entry = fold->at(size - 1);
			fold->erase(fold->begin() + size - 1);
			delete entry;
		}
	}
	int getDimNo() { return dimNo; }
	std::vector<FoldStrain*> *getFold() { return fold; }
	static List<DimensionFold*> *separateDimensionFolds(FoldStrain *foldStrain);
	void print(std::ostream &stream);
};

#endif
