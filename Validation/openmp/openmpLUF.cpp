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

int mainOmpLUF() {

	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension uDims[2];
	Dimension lDims[2];
	Dimension pDims[1];
	pDims[0] = uDims[0] = lDims[0] = aDims[0];
	uDims[1] = lDims[1] = aDims[1];

	// declare new arrays for computation and initialize them
	int uSize = uDims[0].length * uDims[1].length;
	double *nU = new double[uSize];
	for (int i = 0; i < uSize; i++) nU[i] = 0;
	int lSize = lDims[0].length * lDims[1].length;
	double *nL = new double[lSize];
	for (int i = 0; i < lSize; i++) nL[i] = 0;
	int *nP = new int[pDims[0].length];
	for (int i = 0; i < pDims[0].length; i++) nP[i] = 0;

	// starting execution timer clock
	struct timeval start;
	gettimeofday(&start, NULL);

	// declaring common shared variables
	int pivot, nthreads;
	int rows = aDims[0].length;
	int cols = aDims[1].length;

	//-------------------------------- execute LU factorization
	#pragma omp parallel default(shared)
	{
		// prepare step
		#pragma omp for nowait schedule(static)
		for (int i = 0; i < rows; i++) {
			int aRow = i * cols;
			for (int j = 0; j < cols; j++) {
				int uRow = j * cols;
				nU[uRow + i] = a[aRow + j];
			}
		}
		#pragma omp for schedule(static)
		for (int i = 0; i < rows; i++) {
			nL[i * cols + i] = 1;
		}

		// repeat loop
		for (int k = 0; k < rows; k++) {
			// select pivot step
			#pragma omp single
			{
				double max = nU[k * cols + k];
				pivot = k;
				int kRow = k * cols;
				for (int j = k; j < cols; j++) {
					if (nU[kRow + j] > max) {
						max = nU[kRow + j];
						pivot = j;
					}
				}
			}
			// store pivot step
			#pragma omp master
			{
				nP[k] = pivot;
			}
			// interchange columns step
			if (k != pivot) {
				#pragma omp for nowait
				for (int i = k; i < rows; i++) {
					double pivotEntry = nU[i * cols + k];
					nU[i * cols + k] = nU[i * cols + pivot];
					nU[i * cols + pivot] = pivotEntry;
				}
				#pragma omp for
				for (int i = 0; i < k; i++) {
					double pivotEntry = nL[i * cols + k];
					nL[i * cols + k] = nL[i * cols + pivot];
					nL[i * cols + pivot] = pivotEntry;
				}
			}
			// update lower step
			#pragma omp single
			{
				for (int j = k + 1; j < cols; j++) {
					nL[k * cols + j] = nU[k * cols + j] / nU[k * cols + k];
				}
			}
			// update upper step
			#pragma omp for schedule(dynamic)
			for (int i = k; i < rows; i++) {
				for (int j = k + 1; j < cols; j++) {
					nU[i * cols + j] = nU[i * cols + j] - nL[k * cols + j] * nU[i * cols + k];
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
	std::cout << "OpenMP Threads Count: " << nthreads << std::endl;

	return 0;
}
