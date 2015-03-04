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
#include "utils.h"
#include "structures.h"
#include "fileUtility.h"

int mainLUFR() {

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

	//-------------------------------- execute LU factorization sequentially on new arrays
		// prepare step
		for (int i = 0; i < aDims[0].length; i++) {
			int aRow = i * aDims[1].length;
			for (int j = 0; j < aDims[1].length; j++) {
				int uRow = j * uDims[1].length;
				nU[uRow + i] = a[aRow + j];
			}
		}
		for (int i = 0; i < lDims[0].length; i++) {
			int cols = lDims[1].length;
			nL[i * cols + i] = 1;
		}

		// repeat loop
		for (int k = 0; k < aDims[0].length; k++) {

			// select pivot step
			int cols = uDims[1].length;
			double max = nU[k * cols + k];
			int pivot = k;
			int kRow = k * cols;
			for (int j = k; j < uDims[1].length; j++) {
				if (nU[kRow + j] > max) {
					max = nU[kRow + j];
					pivot = j;
				}
			}

			// store pivot step
			nP[k] = pivot;

			// interchange columns step
			if (k != pivot) {
				for (int i = k; i < uDims[0].length; i++) {
					double pivotEntry = nU[i * cols + k];
					nU[i * cols + k] = nU[i * cols + pivot];
					nU[i * cols + pivot] = pivotEntry;
				}
				for (int i = 0; i < k; i++) {
					double pivotEntry = nL[i * cols + k];
					nL[i * cols + k] = nL[i * cols + pivot];
					nL[i * cols + pivot] = pivotEntry;
				}
			}

			// update lower step
			for (int j = k + 1; j < lDims[1].length; j++) {
				nL[k * cols + j] = nU[k * cols + j] / nU[k * cols + k];
			}

			// update upper step
			for (int i = k; i < uDims[0].length; i++) {
				for (int j = k + 1; j < uDims[1].length; j++) {
					nU[i * cols + j] = nU[i * cols + j] - nL[k * cols + j] * nU[i * cols + k];
				}
			}
		}

		//-------------------------------- calculate running time
		struct timeval end;
		gettimeofday(&end, NULL);
		double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
				- (start.tv_sec + start.tv_usec / 1000000.0));
		std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;
}
