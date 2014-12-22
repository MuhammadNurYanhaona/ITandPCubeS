#ifndef _H_loop_index
#define H_loop_index

#include "../utils/list.h"
#include "../utils/hashtable.h"

/*	Classes in this header files are used to determine which index of a parallel loop iteration maps to 
	what array variables and their dimensions. A minimal scope management is needed for this task. Thus
	an IndexScope class is defined along with the IndexArrayAssociation class.  
*/

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
};

class IndexScope {
  protected:
	Hashtable<List<IndexArrayAssociation*>*> *mappings;
	IndexScope *parent;
  public:
	void deriveNewScope();
	void goBackToOldScope();
	static IndexScope *currentScope;
	List<IndexArrayAssociation*> *getAssociationsForArray(const char *array);
	List<IndexArrayAssociation*> *getAssociationsForIndex(const char *index);
	void initiateAssociationList(const char *index);
	IndexScope *getScopeForAssociation(const char *index);
	void saveAssociation(IndexArrayAssociation *association);
  private:
	IndexScope();	
};

#endif
