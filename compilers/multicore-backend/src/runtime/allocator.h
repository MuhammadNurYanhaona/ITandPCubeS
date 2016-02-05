#ifndef _H_allocator
#define _H_allocator

#include "structure.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

namespace allocate {

	// allocates a possibly multidimensional array containing arbitrary type of object as a
	// single dimensional array
	template <class type> type *allocateArray(int dimensionCount, Dimension *dimensions) {
		int length = 1;
		for (int i = 0; i < dimensionCount; i++) {
			length *= dimensions[i].getLength();
		}
		return new type[length];
	}

	// zero fills a possibly multidimensional array of arbitrary type; a zero value is provided
	// for non-primitive types default initial value may not be just zero
	template <class type> void zeroFillArray(type zeroValue, type *array,
			int dimensionCount, Dimension *dimensions) {
		int length = 1;
		for (int i = 0; i < dimensionCount; i++) {
			length *= dimensions[i].getLength();
		}
		for (int i = 0; i < length; i++) {
			array[i] = zeroValue;
		}
	}

	// randomly initialize a possibly multidimensional array of primitive types
	template <class type> void randomFillPrimitiveArray(type *array,
			int dimensionCount, Dimension *dimensions) {
		int length = 1;
		for (int i = 0; i < dimensionCount; i++) {
			length *= dimensions[i].getLength();
		}
		srand(time(NULL));
		for (int i = 0; i < length; i++) {
			array[i] = (type) ((rand() % 100) / 75.00f);
		}
	}
}

#endif
