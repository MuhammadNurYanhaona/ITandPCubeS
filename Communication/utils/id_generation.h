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


#endif /* ID_GENERATION_H_ */
