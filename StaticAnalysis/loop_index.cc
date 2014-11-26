#include "list.h"
#include "hashtable.h"
#include "loop_index.h"

//----------------------------------------------- Index Array Association ------------------------------------------/

IndexArrayAssociation::IndexArrayAssociation(const char *index, const char *array, int dimensionNo) {
	this->index = index;
	this->array = array;
	this->dimensionNo = dimensionNo;
}

//----------------------------------------------------- Index Scope ------------------------------------------------/


IndexScope::IndexScope() {
	mappings = new Hashtable<List<IndexArrayAssociation*>*>;
	parent = NULL;
}	

IndexScope *IndexScope::currentScope = new IndexScope();
	
void IndexScope::goBackToOldScope() { currentScope = parent; }

void IndexScope::deriveNewScope() {
	IndexScope *newScope = new IndexScope();
	newScope->parent = this;
	currentScope = newScope;
}

void IndexScope::initiateAssociationList(const char *index) {
	List<IndexArrayAssociation*> *currentList = mappings->Lookup(index);
	if (currentList != NULL) return;
	currentList = new List<IndexArrayAssociation*>;
	mappings->Enter(index, currentList, true);
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
