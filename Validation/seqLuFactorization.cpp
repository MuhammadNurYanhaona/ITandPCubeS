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

int mainLU() {

	Dimension aDims[2];
	float *a = readArrayFromFile <float> ("a", 2, aDims);
	Dimension uDims[2];
	Dimension lDims[2];
	Dimension pDims[1];
	pDims[0] = uDims[0] = lDims[0] = aDims[0];
	uDims[1] = lDims[1] = aDims[1];

	// declare new arrays for computation and initialize them
	int uSize = uDims[0].length * uDims[1].length;
	float *nU = new float[uSize];
	for (int i = 0; i < uSize; i++) nU[i] = 0;
	int lSize = lDims[0].length * lDims[1].length;
	float *nL = new float[lSize];
	for (int i = 0; i < lSize; i++) nL[i] = 0;
	int *nP = new int[pDims[0].length];
	for (int i = 0; i < pDims[0].length; i++) nP[i] = 0;

	// start timer
	clock_t begin = clock();

	//-------------------------------- execute LU factorization sequentially on new arrays
	// prepare step
	for (int i = 0; i < uSize; i++) nU[i] = a[i];
	for (int i = 0; i < lDims[0].length; i++) {
		int cols = lDims[1].length;
		nL[i * cols + i] = 1;
	}
	// repeat loop
	for (int k = 0; k < aDims[0].length; k++) {

		// select pivot step
		int cols = uDims[1].length;
		float max = nU[k * cols + k];
		int pivot = k;
		for (int i = k; i < uDims[0].length; i++) {
			if (nU[i * cols + k] > max) {
				max = nU[i * cols + k];
				pivot = i;
			}
		}

		// store pivot step
		nP[k] = pivot;

		// interchange rows step
		if (k != pivot) {
			for (int j = k; j < uDims[1].length; j++) {
				float pivotEntry = nU[k * cols + j];
				nU[k * cols + j] = nU[pivot * cols + j];
				nU[pivot * cols + j] = pivotEntry;
			}
			for (int j = 0; j < k; j++) {
				float pivotEntry = nL[k * cols + j];
				nL[k * cols + j] = nL[pivot * cols + j];
				nL[pivot * cols + j] = pivotEntry;
			}
		}

		// update lower step
		for (int i = k + 1; i < lDims[0].length; i++) {
			nL[i * cols + k] = nU[i * cols + k] / nU[k * cols + k];
		}

		// update upper step
		for (int i = k + 1; i < uDims[0].length; i++) {
			for (int j = k; j < uDims[1].length; j++) {
				nU[i * cols + j] = nU[i * cols + j] - nL[i * cols + k] * nU[k * cols + j];
			}
		}
	}

	//-------------------------------- calculate running time
	clock_t end = clock();
	double runningTime = (end - begin) / CLOCKS_PER_SEC;
	std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;
}




