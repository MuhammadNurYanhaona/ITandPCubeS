#ifndef PART_FOLDING_H_
#define PART_FOLDING_H_

#include "../utils/list.h"
#include "../structure.h"
#include <cstdlib>
#include <iostream>

/* This class represents a folding of part IDs from a part-container for any data structure. The folding can
 * subsequently be used to construct a compact interval set representation of the entire container without. */
class PartFolding {
protected:
	int dimNo;
	int level;
	List<PartFolding*> *descendants;
	Range idRange;
public:
	PartFolding(int id, int dimNo, int level) {
		this->dimNo = dimNo;
		this->level = level;
		this->idRange.min = id;
		this->idRange.max = id;
		descendants = new List<PartFolding*>;
	}
	PartFolding(Range *idRange, int dimNo, int level) {
		this->dimNo = dimNo;
		this->level = level;
		this->idRange.min = idRange->min;
		this->idRange.max = idRange->max;
		descendants = new List<PartFolding*>;
	}
	~PartFolding() {
			while (descendants->NumElements() > 0) {
				PartFolding *child = descendants->Nth(0);
				descendants->RemoveAt(0);
				delete child;
			}
			delete descendants;
	}
	bool isEqualInContent(PartFolding *other) {
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
	// coalesce should only be done if two sub-foldings have the same content
	void coalesce(Range otherIdRange) {
		this->idRange.max = otherIdRange.max;
	}
	void addDescendant(PartFolding *descendant) {
		descendants->Append(descendant);
	}
	List<PartFolding*> *getDescendants() { return descendants; }
	Range getIdRange() { return idRange; }
	void print(std::ostream &stream, int indentLevel) {
		stream << '\n';
		for (int i = 0; i < indentLevel; i++) stream << '\t';
		stream << idRange.min << "-" << idRange.max;
		for (int i = 0; i < descendants->NumElements(); i++) {
			descendants->Nth(i)->print(stream, indentLevel + 1);
		}
	}
};

#endif
