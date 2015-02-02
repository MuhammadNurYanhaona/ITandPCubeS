#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "loop_index.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_type.h"
#include "../syntax/ast.h"
#include "../semantics/task_space.h"

#include <iostream>
#include <sstream>

//----------------------------------------------- Index Array Association ------------------------------------------/

IndexArrayAssociation::IndexArrayAssociation(const char *index, const char *array, int dimensionNo) {
	this->index = index;
	this->array = array;
	this->dimensionNo = dimensionNo;
}

bool IndexArrayAssociation::isEqual(IndexArrayAssociation *other) {
	return strcmp(this->index, other->index) == 0 
			&& strcmp(this->array, other->array) == 0
			&& this->dimensionNo == other->dimensionNo;
}

List<IndexArrayAssociation*> *IndexArrayAssociation::filterList(List<IndexArrayAssociation*> *list) {
	if (list->NumElements() == 0) return list;
	List<IndexArrayAssociation*> *filteredList = new List<IndexArrayAssociation*>;
	for (int i = 0; i < list->NumElements(); i++) {
		IndexArrayAssociation *current = list->Nth(i);
		bool found = false;
		for (int j = 0; j < filteredList->NumElements(); j++) {
			if (current->isEqual(filteredList->Nth(j))) {
				found = true;
				break;
			}
		}
		if (!found) filteredList->Append(current);
	}
	return filteredList;
}

RangeExpr *IndexArrayAssociation::convertToRangeExpr(Type *arrayType) {

	Identifier *id = new Identifier(yylloc, index);
	DimensionIdentifier *dimensionId = new DimensionIdentifier(yylloc, dimensionNo + 1);
	FieldAccess *arrayReference = new FieldAccess(NULL, new Identifier(yylloc, array), yylloc);
	arrayReference->setMetadata(true);
	arrayReference->markLocal();
	arrayReference->setType(arrayType);
	FieldAccess *dimensionAccess = new FieldAccess(arrayReference, dimensionId, yylloc);
	Identifier *rangeId = new Identifier(yylloc, "range");
	FieldAccess *rangeAccess = new FieldAccess(dimensionAccess, rangeId, yylloc);
	return new RangeExpr(id, rangeAccess, NULL, true, yylloc);
}

void IndexArrayAssociation::generateTransform(std::ostringstream &stream, int indentLevel, Space *space) {
	
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';

	ArrayDataStructure *structure = (ArrayDataStructure*) space->getStructure(array);
	int dimensionCount = structure->getDimensionality();
	std::ostringstream xform;
	xform << "(" << index;
	// TODO the following line is disabled to make code pleasing to look at for manual inspection. Once
	// we are done with debugging, we have to enable it.
	// xform << " - " << array << "PartDims[" << dimensionNo << "].getPositiveRange().min";
	xform << ")";
	bool firstEntry = true;
	for (int i = dimensionCount - 1; i > dimensionNo; i--) {
		if (!firstEntry) {
			xform << '\n' << indent.str() << "\t\t";
		}
		xform << " * " << array << "StoreDims[" << i << "].getLength()";
		firstEntry = false;
	}
	stream << indent.str();
	stream << "int " << index << array << dimensionNo;
	stream << " = " << xform.str() << ";\n";
}

//----------------------------------------------------- Index Scope ------------------------------------------------/

IndexScope::IndexScope() {
	mappings = new Hashtable<List<IndexArrayAssociation*>*>;
	parent = NULL;
	orderedIndexList = new List<const char*>;
	preferredArrayForIndexTraversal = new Hashtable<const char*>;
}	

IndexScope *IndexScope::currentScope = new IndexScope();
	
void IndexScope::goBackToOldScope() { currentScope = parent; }

void IndexScope::deriveNewScope() {
	IndexScope *newScope = new IndexScope();
	newScope->parent = this;
	currentScope = newScope;
}

void IndexScope::enterScope(IndexScope *newScope) {
	newScope->parent = this;
	currentScope = newScope;
}

void IndexScope::initiateAssociationList(const char *index) {
	List<IndexArrayAssociation*> *currentList = mappings->Lookup(index);
	if (currentList != NULL) return;
	currentList = new List<IndexArrayAssociation*>;
	mappings->Enter(index, currentList, true);
	orderedIndexList->Append(index);
}

IndexScope *IndexScope::getScopeForAssociation(const char *index) {
	if (mappings->Lookup(index) != NULL) return this;
	if (parent == NULL) return NULL;
	return parent->getScopeForAssociation(index);
}

void IndexScope::saveAssociation(IndexArrayAssociation *association) {
	List<IndexArrayAssociation*> *currentList = mappings->Lookup(association->getIndex());
	if (currentList == NULL) {
		printf("Internal error while trying to associate index with arrays in parallel for loops");
		return;
	}
	currentList->Append(association);
	mappings->Enter(association->getIndex(), currentList, true);
}

List<IndexArrayAssociation*> *IndexScope::getAssociationsForArray(const char *array) {
	List<IndexArrayAssociation*> *associationList = new List<IndexArrayAssociation*>;
	Iterator<List<IndexArrayAssociation*>*> iter = mappings->GetIterator();
	IndexArrayAssociation *association;
	List<IndexArrayAssociation*> *list;
	while ((list = iter.GetNextValue()) != NULL) {
		for (int i = 0; i < list->NumElements(); i++) {
			association = list->Nth(i);	
			if (strcmp(association->getArray(), array) == 0) {
				associationList->Append(association);
			}
		}
	}
	return associationList;	
}

List<IndexArrayAssociation*> *IndexScope::getAssociationsForIndex(const char *index) {
	IndexScope *indexScope = getScopeForAssociation(index);
	if (indexScope == NULL) return NULL;
	return indexScope->mappings->Lookup(index);
}

void IndexScope::setPreferredArrayForIndex(const char *index, const char *array) {
	preferredArrayForIndexTraversal->Enter(index, array, true);
}

IndexArrayAssociation *IndexScope::getPreferredAssociation(const char *index) {
	
	const char *preferredArray = preferredArrayForIndexTraversal->Lookup(index);
	List<IndexArrayAssociation*> *associationList = getAssociationsForIndex(index);
	
	if (associationList == NULL 
		|| associationList->NumElements() == 0) return NULL;
	if (preferredArray == NULL) return associationList->Nth(0);

	for (int i = 0; i < associationList->NumElements(); i++) {
		IndexArrayAssociation *association = associationList->Nth(i);
		if (strcmp(association->getArray(), preferredArray) == 0) {
			return association;
		}
	}
	return associationList->Nth(0);
}

List<IndexArrayAssociation*> *IndexScope::getAllPreferredAssociations() {
	List<IndexArrayAssociation*> *preferredList = new List<IndexArrayAssociation*>;
	for (int i = 0; i < orderedIndexList->NumElements(); i++) {
		preferredList->Append(getPreferredAssociation(orderedIndexList->Nth(i)));
	}
	return preferredList; 
}
