#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"

int mainMMMV() {

	// load original inputs and output from generated files
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension bDims[2];
	double *b = readArrayFromFile <double> ("b", 2, bDims);
	Dimension cDims[2];
	double *cOrig = readArrayFromFile <double> ("c", 2, cDims);

	// declare and initialize c for current computation
	int cSize = aDims[0].length * bDims[1].length;
	double *c = new double[cSize];
	for (int i = 0; i < cSize; i++) c[i] = 0;

	// calculate new C
	for (int i = 0; i < aDims[0].length; i++) {
		int aRowIndex = i * aDims[1].length;
		int cRowIndex = i * cDims[1].length;
		for (int j = 0; j < bDims[1].length; j++) {
			for (int k = 0; k < aDims[1].length; k++) {
				int bRowIndex = k * bDims[1].length;
				c[cRowIndex + j] += a[aRowIndex + k] * b[bRowIndex + j];
			}
		}
	}

	// compare new C with original C
	bool fault = false;
	for (int i = 0; i < cSize; i++) {
		if (abs(c[i] - cOrig[i]) > 0.1) {
			std::cout << "computed C and original C did not match at ";
			std::cout << "index " << i << " C = " << c[i] << " but ";
			std::cout << "original C = " << cOrig[i] << std::endl;
			std::cout << "Difference is: " << c[i] - cOrig[i] << std::endl;
			fault = true;
		}
	}

	if (!fault) {
		std::cout << "validation successful\n";
	}

	return 0;
}



