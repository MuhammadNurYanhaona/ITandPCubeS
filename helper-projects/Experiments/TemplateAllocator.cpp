#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include "utils.h"
#include "structures.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>

template <class type> type *allocateArray(int dimensionCount, Dimension *dimensions) {
	int length = 1;
	for (int i = 0; i < dimensionCount; i++) {
		length *= dimensions[i].length;
	}
	return new type[length];
}

template <class type> void zeroFillArray(type zeroValue, type *array,
				int dimensionCount, Dimension *dimensions) {
	int length = 1;
	for (int i = 0; i < dimensionCount; i++) {
			length *= dimensions[i].length;
	}
	std::cout << "Total length: " << length << std::endl;
	for (int i = 0; i < length; i++) {
		array[i] = zeroValue;
	}
}

template <class type> void randomFillPrimitiveArray(type *array,
				int dimensionCount, Dimension *dimensions) {
	int length = 1;
	for (int i = 0; i < dimensionCount; i++) {
		length *= dimensions[i].length;
	}
	srand(time(NULL));
	for (int i = 0; i < length; i++) {
		array[i] = (type) rand();
	}
}

int mainAllocate() {
	Dimension matrix[2];
	matrix[0].length = 5;
	matrix[1].length = 2;
	int *array = allocateArray <int> (2, matrix);
	randomFillPrimitiveArray <int> (array, 2, matrix);
	for (int i = 0; i < matrix[0].length; i++) {
		int beginIndex = i * matrix[1].length;
		for (int j = 0; j < matrix[1].length; j++) {
			std::cout << array[beginIndex + j] << "\t";
		}
		std::cout << std::endl;
	}
	return 0;
}



