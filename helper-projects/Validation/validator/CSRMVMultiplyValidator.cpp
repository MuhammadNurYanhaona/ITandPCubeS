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

int mainCSRMVMV() {

	// reading elements of the CSR sparse matrix
	Dimension colDims[1];
	int *columns = readArrayFromFile <int> ("columns", 1, colDims);
	Dimension rowDims[1];
	int *rows = readArrayFromFile <int> ("rows", 1, rowDims);
	Dimension valDims[1];
	double *values = readArrayFromFile <double> ("values", 1, valDims);
	// reading input vector
	Dimension vDims[1];
	double *v = readArrayFromFile <double> ("v", 1, vDims);
	// reading the solution vector
	Dimension wDims[1];
	double *w = readArrayFromFile <double> ("w", 1, wDims);

	// declare and initialize the computed result vector
	double *wCom = new double[wDims[0].length];
	for (int i = 0; i < wDims[0].length; i++) wCom[i] = 0;

	// do the computation
	for (int i = 0; i < rowDims[0].length; i++) {
		int start = (i > 0) ? rows[i - 1] + 1 : 0;
		int end = rows[i];
		for (int j = start; j <= end; j++) {
			wCom[i] = wCom[i] + values[j] * v[columns[j]];
		}
	}

	// compare results
	bool valid = true;
	for (int i = 0; i < wDims[0].length; i++) {
		if (abs(w[i] - wCom[i]) > 0.1) {
			std::cout << "Computed w did not match at index [" << i << "]\n";
			std::cout << "Computed: " << wCom[i] << " Read: " << w[i] << "\n";
			valid = false;
		}
	}
	if (valid) std::cout << "validation successful\n";

	return 0;
}



