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

int mainPLUFV() {

	// read all arrays from file
	Dimension aDims[2];
	double *a = readArrayFromFile <double> ("a", 2, aDims);
	Dimension uDims[2];
	double *u = readArrayFromFile <double> ("u", 2, uDims);
	Dimension lDims[2];
	double *l = readArrayFromFile <double> ("l", 2, lDims);
	Dimension pDims[1];
	int *p = readArrayFromFile <int> ("p", 1, pDims);

	int blockSize = 32;

	// declare new arrays for computation and initialize them
	int uSize = uDims[0].length * uDims[1].length;
	double *nU = new double[uSize];
	for (int i = 0; i < uSize; i++) nU[i] = 0;
	int lSize = lDims[0].length * lDims[1].length;
	double *nL = new double[lSize];
	for (int i = 0; i < lSize; i++) nL[i] = 0;
	int *nP = new int[pDims[0].length];
	for (int i = 0; i < pDims[0].length; i++) nP[i] = 0;
	double *p_column = new double[uDims[0].length];
	for (int i = 0; i < uDims[0].length; i++) p_column[i] = 0;

	// Execute code equivalent to Initialize LU task
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

	Range rows = aDims[0].range;
	int max1 = rows.max;
	int max2 = aDims[1].range.max;

	for (int k = 0; k <= rows.max; k += blockSize) {
		int lastRow = k + blockSize - 1;
		if (lastRow > max1) lastRow = max1;
		Range range;
		range.min = k;
		range.max = lastRow;

		// execute code equivalent to Transposed LU Factorization
		{
			// repeat loop
			for (int k = range.min; k <= range.max; k++) {

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
					nU[k * cols + j] = 0;
				}

				// update upper step
				for (int i = k + 1; i <= range.max; i++) {
					for (int j = k + 1; j < uDims[1].length; j++) {
						nU[i * cols + j] = nU[i * cols + j] - nL[k * cols + j] * nU[i * cols + k];
					}
				}

				// retrieve lower part step
				for (int i = range.min; i < k; i++) {
					p_column[i] = nL[i * cols + k];
				}

				// update upper column step
				for (int i = range.max + 1; i < uDims[0].length; i++) {
					for (int j = range.min; j < k; j++) {
						nU[i * cols + k] = nU[i * cols + k] - nU[i * cols + j] * p_column[j];
					}
				}
			}
		}

		if (lastRow < max1) {
			// execute code equivalent to Subtract Matrix Multiply Result
			int cols = uDims[1].length;
			for (int iB = lastRow + 1; iB <= max1; iB += blockSize) {
				int rStart = iB;
				int rEnd = rStart + blockSize - 1;
				if (rEnd > max1) rEnd = max1;
				for (int jB = lastRow + 1; jB <= max2; jB += blockSize) {
					int cStart = jB;
					int cEnd = cStart + blockSize - 1;
					if (cEnd > max2) cEnd = max2;
					for (int kB = k; kB <= lastRow; kB += blockSize) {
						int startIndex = kB;
						int endIndex = startIndex + blockSize - 1;
						if (endIndex > lastRow) endIndex = lastRow;
						for (int i = rStart; i <= rEnd; i++) {
							for (int j = cStart; j <= cEnd; j++) {
								double total = 0.0;
								for (int l = startIndex; l <= endIndex; l++) {
									total += nU[i * cols + l] * nL[l * cols + j];
								}
								nU[i * cols + j] -= total;
							}
						}
					}
				}
			}

		}
	}

	//------------------------------------------------- finally, check if all arrays match
	bool valid = true;
	int invalid = 0;
	for (int i = 0; i < pDims[0].length; i++) {
		if (nP[i] != p[i]) {
			invalid++;
			valid = false;
		}
	}
	if (!valid) std::cout << "P does not match " << invalid << " times\n";
	invalid = 0;
	for (int i = 0; i < uSize; i++) {
		if (abs(u[i] - nU[i]) > 0.1) {
			invalid++;
			valid = false;
		}
	}
	if (!valid) std::cout << "u does not match " << invalid << " times\n";
	invalid = 0;
	for (int i = 0; i < lSize; i++) {
		if (abs(l[i] - nL[i]) > 0.1) {
			invalid++;
			valid = false;
		}
	}
	if (!valid) std::cout << "l does not match " << invalid << " times\n";
	if (valid) std::cout << "validation successful\n";

	return 0;
}



