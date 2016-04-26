#ifndef ID_GENERATION_H_
#define ID_GENERATION_H_

#include "list.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace idutils {

	inline List<int*> *generateIdFromArray(int *id, int dimensions, int length) {
		List<int*> *listId = new List<int*>;
		int levels = length /dimensions;
		for (int i = 0; i < levels; i++) {
			listId->Append(id + i * dimensions);
		}
		return listId;
	}

	inline bool areIdsEqual(List<int*> *id1, List<int*> *id2, int idDimensions) {
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

	inline bool areIdsEqual(std::vector<int*> *id1, std::vector<int*> *id2, int idDimensions) {
		if (id1->size() != id2->size()) return false;
		for (unsigned int level = 0; level < id1->size(); level++) {
			int *id1AtLevel = id1->at(level);
			int *id2AtLevel = id2->at(level);
			for (int dimension = 0; dimension < idDimensions; dimension++) {
				if (id1AtLevel[dimension] != id2AtLevel[dimension]) return false;
			}
		}
		return true;
	}

	inline int concateIds(int id1, int id2, int digitCount) {
		int result;
		std::ostringstream ostream;
		ostream << std::setfill('0') << std::setw(digitCount) << id1;
		ostream << std::setfill('0') << std::setw(digitCount) << id2;
		std::istringstream istream(ostream.str());
		istream >> result;
		return result;
	}
}

#endif /* ID_GENERATION_H_ */
