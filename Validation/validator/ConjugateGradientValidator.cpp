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

int mainCGV() {
	// read all arrays from file
	// reading elements of the CSR sparse matrix
	Dimension colDims[1];
	int *columns = readArrayFromFile <int> ("columns", 1, colDims);
	Dimension rowDims[1];
	int *rows = readArrayFromFile <int> ("rows", 1, rowDims);
	Dimension valDims[1];
	double *values = readArrayFromFile <double> ("values", 1, valDims);
	// reading other vectors
	Dimension bDims[1];
	double *b = readArrayFromFile <double> ("b", 1, bDims);
	Dimension x_iDims[1];
	double *x_i = readArrayFromFile <double> ("x_0", 1, x_iDims);
	// reading the solution vector
	Dimension xDims[1];
	double *x = readArrayFromFile <double> ("x", 1, xDims);

	// declare iteration limiter variables
	const int maxIterations = 10;
	const double precision = 1;

	// declare other variables used in the algorithm
	double *r_i = new double[xDims[0].length];
	double *a_r_i = new double[xDims[0].length];
	double norm = 0;
	int iteration = 0;

	// run the algorithm
	do {
		// calculating r_i = b - A * x_i
		for (int i = 0; i < xDims[0].length; i++) r_i[i] = 0;
		for (int i = 0; i < rowDims[0].length; i++) {
			int start = (i > 0) ? rows[i - 1] + 1 : 0;
			int end = rows[i];
			for (int j = start; j <= end; j++) {
				r_i[i] = r_i[i] + values[j] * x_i[columns[j]];
			}
		}
		for (int i = 0; i < bDims[0].length; i++) {
			r_i[i] = b[i] - r_i[i];
		}

		// calculating alpha_i = (r_i * r_i) / (r_i * (A * r_i))
		for (int i = 0; i < xDims[0].length; i++) a_r_i[i] = 0;
		for (int i = 0; i < rowDims[0].length; i++) {
			int start = (i > 0) ? rows[i - 1] + 1 : 0;
			int end = rows[i];
			for (int j = start; j <= end; j++) {
				a_r_i[i] = a_r_i[i] + values[j] * r_i[columns[j]];
			}
		}
		norm = 0;
		double denominator = 0;
		for (int i = 0; i < xDims[0].length; i++) {
			norm += r_i[i] * r_i[i];
			denominator += r_i[i] * a_r_i[i];
		}
		double alpha_i = norm / denominator;

		// calculating x_i = x_i + alpha_i * r_i
		for (int i = 0; i < xDims[0].length; i++) {
			x_i[i] = x_i[i] + alpha_i * r_i[i];
		}

		iteration++;
	} while (iteration < maxIterations && norm > precision);

	bool valid = true;
	for (int i = 0; i < xDims[0].length; i++) {
		if (abs(x[i] / x_i[i]) > 1.001 || abs(x_i[i] / x[i]) > 1.001) {
			std::cout << "Computed x did not match at index [" << i << "]\n";
			std::cout << "x: " << x[i] << " x_computed: " << x_i[i] << std::endl;
			valid = false;
		}
	}
	if (valid) std::cout << "validation successful\n";
	return 0;
}

