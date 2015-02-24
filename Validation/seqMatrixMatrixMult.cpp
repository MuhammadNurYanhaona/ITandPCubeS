#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <time.h>
#include "utils.h"
#include "structures.h"
#include "fileUtility.h"

int main() {

	// load original inputs and output from generated files
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension bDims[2];
	double *b = readArrayFromFile <double> ("b", 2, bDims);
	Dimension cDims[2];

	// declare and initialize c for current computation
	int cSize = aDims[0].length * bDims[1].length;
	double *c = new double[cSize];
	for (int i = 0; i < cSize; i++) c[i] = 0;

	// start timer
	clock_t begin = clock();

	// execute matrix-matrix multiplication sequentially
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

	//-------------------------------- calculate running time
	clock_t end = clock();
	double runningTime = (end - begin) / CLOCKS_PER_SEC;
	std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;

	return 0;
}
