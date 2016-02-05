#ifndef _H_loop_index
#define H_loop_index

#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <iostream>
#include <sstream>

/*	Classes in this header files are used to determine which index of a parallel loop iteration maps to 
	what array variables and their dimensions. A minimal scope management is needed for this task. Thus
	an IndexScope class is defined along with the IndexArrayAssociation class.  
*/

class RangeExpr;
class Type;
class Space;

class IndexArrayAssociation {
 protected:
	const char *index;
	const char *array;
	int dimensionNo;
 public:
	IndexArrayAssociation(const char *index, const char *array, int dimensionNo);
	const char *getArray() { return array; }
	const char *getIndex() { return index; }
	int getDimensionNo() { return dimensionNo; }
	bool isEqual(IndexArrayAssociation *other);
	static List<IndexArrayAssociation*> *filterList(List<IndexArrayAssociation*> *list);

	// helper methods for code generation
	RangeExpr *convertToRangeExpr(Type *arrayType);
	// This method is been added to aid multidimensional to unidimensional index transform. Doing the
	// transformation here (equivalent to doing it just after the creation of an index loop) instead of
	// doing that inside an array access expression has the benefit of hoisting repetative computations
	// out of nested loops.	
	void generateTransform(std::ostringstream &stream, int indentLevel, Space *space);
};

class IndexScope {
  protected:
	// keeps track of the list of index accessed done on different arrays using a particular index
	Hashtable<List<IndexArrayAssociation*>*> *mappings;
	IndexScope *parent;
	// stores information about the order in which indexes appeared in the source code to determine the
	// loop nesting order for multiply indexed IT loops
	List<const char*> *orderedIndexList;
	// stores the names of arrays whose dimesnions the task will like index iteration bounds to be set
	// based on
	Hashtable<const char*> *preferredArrayForIndexTraversal;
  public:
	void deriveNewScope();
	void enterScope(IndexScope *newScope);
	void goBackToOldScope();
	static IndexScope *currentScope;
	List<IndexArrayAssociation*> *getAssociationsForArray(const char *array);
	List<IndexArrayAssociation*> *getAssociationsForIndex(const char *index);
	void initiateAssociationList(const char *index);
	IndexScope *getScopeForAssociation(const char *index);
	void saveAssociation(IndexArrayAssociation *association);
	void setPreferredArrayForIndex(const char *index, const char *array);
	IndexArrayAssociation *getPreferredAssociation(const char *index);
	// returns the preferred association list for all indexes in the scope in the order prescribed by
	// ordered-index-list entries.
	List<IndexArrayAssociation*> *getAllPreferredAssociations();
	
  private:
	IndexScope();	
};

#endif
