#ifndef _H_binary_search
#define _H_binary_search

#include <vector>

static const int KEY_NOT_FOUND = -1;

namespace binsearch {
	int locateKey(std::vector<int> array, int key) {
		
		int minIndex = 0;
		int maxIndex = array.size() - 1;
		while (maxIndex >= minIndex) {
			int midpoint = minIndex + ((maxIndex - minIndex) / 2);
			if (array[midpoint] == key) {
				return midpoint;
			} else if (array[midpoint] < key) {
				minIndex = midpoint + 1;
			} else {
				maxIndex = midpoint - 1;
			}
		}
		return KEY_NOT_FOUND;
	}

	int locatePointOfInsert(std::vector<int> array, int key) {

		if (array.empty()) return 0;
		int minIndex = 0;
		int maxIndex = array.size() - 1;

		if (array[0] > key) return 0;
		if (array[maxIndex] < key) return array.size();

		while (maxIndex >= minIndex) {
			int midpoint = minIndex + ((maxIndex - minIndex) / 2);
			if (array[midpoint] < key) {
				minIndex = midpoint + 1;
			} else {
				maxIndex = midpoint - 1;
			}
		}
		return maxIndex + 1;
	}
}

#endif
