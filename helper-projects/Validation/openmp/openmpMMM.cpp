#include <omp.h>
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

int mainOmpMMM() {
	// load original inputs and output from generated files
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension bDims[2];
	double *b = readArrayFromFile <double> ("b", 2, bDims);

	// declare and initialize c for current computation
	int cSize = aDims[0].length * bDims[1].length;
	double *c = new double[cSize];
	for (int i = 0; i < cSize; i++) c[i] = 0;

	// starting execution timer clock
	struct timeval start;
	gettimeofday(&start, NULL);

	// get the length information for each dimension
	int rows = aDims[0].length;
	int commons = aDims[1].length;
	int cols = bDims[1].length;

	// declare index variables
	int i, j, k;
	int nthreads;

	// execute matrix-matrix multiplication
	#pragma omp parallel default(shared) private(i,j,k)
	{
		#pragma omp for schedule(static)
		for (i = 0; i < rows; i++) {
			for (j = 0; j < cols; j++) {
				for (k = 0; k < commons; k++) {
					c[i * cols + j] += a[i * commons + k] * b[k * cols + j];
				}
			}
		}
		nthreads = omp_get_num_threads();
	}

	//-------------------------------- calculate running time
	struct timeval end;
	gettimeofday(&end, NULL);
	double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	std::cout << "Execution Time: " << runningTime << " Seconds" << std::endl;
	std::cout << "Number of OpenMP threads = " << nthreads << '\n';

	return 0;
}
