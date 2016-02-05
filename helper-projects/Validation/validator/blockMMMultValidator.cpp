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

int mainBMMMV(int argc, const char *argv[]) {

	// load original inputs and output from generated files
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension bDims[2];
	double *b = readArrayFromFile <double> ("b", 2, bDims);
	Dimension cDims[2];
	double *cOrig = readArrayFromFile <double> ("c", 2, cDims);

	// read the block size from command line
	int blockSize = 32;
	if (argc > 1) blockSize = atoi(argv[1]);

	// declare and initialize c for current computation
	int cSize = aDims[0].length * bDims[1].length;
	double *c = new double[cSize];
	for (int i = 0; i < cSize; i++) c[i] = 0;

	// calculate new C
	for (int iB = 0; iB < aDims[0].length; iB += blockSize) {
		int rStart = iB;
		int rEnd = rStart + blockSize - 1;
		if (rEnd >= aDims[0].length) rEnd = aDims[0].length - 1;
		for (int jB = 0; jB < bDims[1].length; jB += blockSize) {
			int cStart = jB;
			int cEnd = cStart + blockSize - 1;
			if (cEnd >= bDims[1].length) cEnd = bDims[1].length - 1;
			for (int kB = 0; kB < aDims[1].length; kB += blockSize) {
				int startIndex = kB;
				int endIndex = startIndex + blockSize - 1;
				if (endIndex >= aDims[1].length) endIndex = aDims[1].length - 1;
				for (int i = rStart; i <= rEnd; i++) {
					int aRowIndex = i * aDims[1].length;
					int cRowIndex = i * cDims[1].length;
					for (int j = cStart; j <= cEnd; j++) {
						for (int k = startIndex; k <= endIndex; k++) {
							int bRowIndex = k * bDims[1].length;
							c[cRowIndex + j] += a[aRowIndex + k] * b[bRowIndex + j];
						}
					}
				}
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



