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

using namespace std;

double executeVectorDotProduct(Dimension dimension, double *u, double *v);

void executeVectorAddition(Dimension dimension, double *w, double *u, double *v, int alpha, int beta);

void executeCSRMatrixVectorMultiply(Dimension rowDim, Dimension colDim, 
		int *rows, int *columns, double *values, 
		double *v, double *w);

int mainCG(int argc, char *argv[]) {

	if (argc < 3) {
		cout << "pass the maximum number of iterations and desired precision for convergeance test\n";
		exit(EXIT_FAILURE);
	}

	//--------------------------------------------------------------------------- Data Reading

	// load the components of the sparse matrix from files
        Dimension colDims[1];
        int *columns = readArrayFromFile <int> ("sparse matrix columns", 1, colDims);
        Dimension rowDims[1];
        int *rows = readArrayFromFile <int> ("sparse matrix rows", 1, rowDims);
        Dimension valDims[1];
        double *values = readArrayFromFile <double> ("sparse matrix element values", 1, valDims);

	// load the known vector
	Dimension bDims[1];
	double *b = readArrayFromFile <double> ("known vector", 1, bDims);
	// load the prediction vector
	Dimension xDims[1];
	double *x_i = readArrayFromFile <double> ("prediction vector", 1, xDims);


	// starting execution timer clock
        struct timeval start;
        gettimeofday(&start, NULL);
	//------------------------------------------------------------------------- Program Begins
	
	
	// create some additional temporary variables for holding intermediate data
	double *w = (double *) malloc(xDims[0].length * sizeof(double));
	double *r_i = (double *) malloc(xDims[0].length * sizeof(double));
	double *A_r_i = (double *) malloc(xDims[0].length * sizeof(double));

	int maxIterations = atoi(argv[1]);
	double precision = atof(argv[2]);

	// do the specified number of iterations (for now ignore convergence precision)	
	int iteration = 0;
	for (iteration = 0; iteration < maxIterations; iteration++) {
		
		// calculate w = A * x_i
		executeCSRMatrixVectorMultiply(rowDims[0], colDims[0],
                		rows, columns, values, x_i, w);	

		// determine the current residual error as r_i = b - A * x_i
		executeVectorAddition(xDims[0], r_i, b, w, 1, -1);
		
		// determine the dot product of r_i to itself as the residual norm
		double norm = executeVectorDotProduct(xDims[0], r_i, r_i);

		// determine A * r_i
		executeCSRMatrixVectorMultiply(rowDims[0], colDims[0],
                                rows, columns, values, r_i, A_r_i);

		// determine dot product of r_i to A * r_i
		double product = executeVectorDotProduct(xDims[0], r_i, A_r_i);
	
		// determine the next step size alpha_i as (r_i.r_i) / (r_i.(A * r_i))
                double alpha_i = norm / product;
		
		// calculate the next estimate x_i = x_i + alpha_i * r_i
		executeVectorAddition(xDims[0], w, x_i, r_i, 1, alpha_i);
		for (int i = xDims[0].range.min; i <= xDims[0].range.max; i++) {
			x_i[i] = w[i];
		}
	}
	
	//--------------------------------------------------------------------------- Program Ends
	struct timeval end;
        gettimeofday(&end, NULL);
        double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
                        - (start.tv_sec + start.tv_usec / 1000000.0));
        std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;

	return 0;
}

void executeCSRMatrixVectorMultiply(Dimension rowDim, Dimension colDim, 
		int *rows, int *columns, double *values, 
		double *v, double *w) {

	for (int i = 0; i <= rowDim.range.max; i++) {
		int startColumn = 0;
		if (i > 0) {
			startColumn = rows[i - 1] + 1;
		}
		int endColumn = rows[i];
		for (int j = startColumn; j <= endColumn; j++) {
			w[i] += values[j] * v[columns[j]];
		}
	}
}

void executeVectorAddition(Dimension dimension, double *w, double *u, double *v, int alpha, int beta) {
	
	for (int i = dimension.range.min; i <= dimension.range.max; i++) {
		w[i] = alpha * u[i] + beta * v[i];
	}
}

double executeVectorDotProduct(Dimension dimension, double *u, double *v) {
	
	double product = 0;
	for (int i = dimension.range.min; i <= dimension.range.max; i++) {
		product += u[i] * v[i];
	}
	return product;
}
