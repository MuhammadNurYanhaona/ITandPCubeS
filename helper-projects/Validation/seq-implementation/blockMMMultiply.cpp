#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <sys/time.h>
#include <time.h>
#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"

int mainBMMM(int argc, const char *argv[]) {

	// load original inputs and output from generated files
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension bDims[2];
	double *b = readArrayFromFile <double> ("b", 2, bDims);
	Dimension cDims[2];
	cDims[0] = aDims[0]; cDims[1] = bDims[1];

	// read the block size from command line
	int blockSize = 32;
	if (argc > 1) blockSize = atoi(argv[1]);

	// starting execution timer clock
	struct timeval start;
	gettimeofday(&start, NULL);

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

	//-------------------------------- calculate running time
	struct timeval end;
	gettimeofday(&end, NULL);
	double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;

	return 0;
}



