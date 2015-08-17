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

int mainLUFVC() {

	// read all arrays from file
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension uDims[2];
	double *u = readArrayFromFile <double> ("u", 2, uDims);
	Dimension lDims[2];
	double *l = readArrayFromFile <double> ("l", 2, lDims);
	Dimension pDims[1];
	int *p = readArrayFromFile <int> ("p", 1, pDims);

	// declare new arrays for computation and initialize them
	int uSize = uDims[0].length * uDims[1].length;
	double *nU = new double[uSize];
	for (int i = 0; i < uSize; i++) nU[i] = 0;
	int lSize = lDims[0].length * lDims[1].length;
	double *nL = new double[lSize];
	for (int i = 0; i < lSize; i++) nL[i] = 0;
	int *nP = new int[pDims[0].length];
	for (int i = 0; i < pDims[0].length; i++) nP[i] = 0;

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
		double max = nU[k * cols + k];
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
				double pivotEntry = nU[k * cols + j];
				nU[k * cols + j] = nU[pivot * cols + j];
				nU[pivot * cols + j] = pivotEntry;
			}
			for (int j = 0; j < k; j++) {
				double pivotEntry = nL[k * cols + j];
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

	//------------------------------------------------- finally, check if all arrays match
	bool valid = true;
	for (int i = 0; i < pDims[0].length; i++) {
		if (nP[i] != p[i]) {
			std::cout << "Computed P did not match at index [" << i << "]\n";
			valid = false;
		}
	}
	for (int i = 0; i < uSize; i++) {
		if (abs(u[i] - nU[i]) > 0.1) {
			int row = i / uDims[1].length;
			int cols = i - row * uDims[1].length;
			std::cout << "Computed U did not match at index [" << row << "][" << cols << "]\n";
			valid = false;
		}
	}
	for (int i = 0; i < lSize; i++) {
		if (abs(l[i] - nL[i]) > 0.1) {
			int row = i / lDims[1].length;
			int cols = i - row * lDims[1].length;
			std::cout << "Computed L did not match at index [" << row << "][" << cols << "]\n";
			valid = false;
		}
	}
	if (valid) std::cout << "validation successful\n";

	return 0;
}



