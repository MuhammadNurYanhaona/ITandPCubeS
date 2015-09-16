#ifndef ID_GENERATION_H_
#define ID_GENERATION_H_

#include "list.h"

inline List<int*> *generateIdFromArray(int *id, int dimensions, int length) {
	List<int*> *listId = new List<int*>;
	int levels = length /dimensions;
	for (int i = 0; i < levels; i++) {
		listId->Append(id + i * dimensions);
	}
	return listId;
}

inline bool isIdsEqual(List<int*> *id1, List<int*> *id2, int idDimensions) {
	if (id1->NumElements() != id2->NumElements()) return false;
	for (int level = 0; level < id1->NumElements(); level++) {
		int *id1AtLevel = id1->Nth(level);
		int *id2AtLevel = id2->Nth(level);
		for (int dimension = 0; dimension < idDimensions; dimension++) {
			if (id1AtLevel[dimension] != id2AtLevel[dimension]) return false;
		}
	}
	return true;
}


#endif /* ID_GENERATION_H_ */
