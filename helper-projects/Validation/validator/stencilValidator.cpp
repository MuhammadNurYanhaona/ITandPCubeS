#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <string.h>
#include <deque>
#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"

using namespace std;

int mainSV() {

	// read the initial plate state from a file
	Dimension plateDims[2];
	double *plate0 = readArrayFromFile <double> ("plate initial state", 2, plateDims);
	// read the parallel code's final plate state output from another file
	double *plateOut = readArrayFromFile <double> ("plate final state", 2, plateDims);

	// specify the number of refinement iterations
	cout << "Enter the number of refinement iterations needed\n";
	int iterations;
	cin >> iterations;

	// allocate another plate variable for computation
	int plateSize = plateDims[0].length * plateDims[1].length;
	double *plate1 = new double[plateSize];

	// synchronize the alternative version of the plate with the plate read from the file
	for (int i = 0; i < plateSize; i++) {
		plate1[i] = plate0[i];
	}

	// do the iterative stencil computation
	double *oldPlate, *newPlate;
	oldPlate = NULL;
	newPlate = NULL;
	for (int i = 0; i < iterations; i++) {
		if (i % 2 == 0) {
			oldPlate = plate0;
			newPlate = plate1;
		} else {
			oldPlate = plate1;
			newPlate = plate0;
		}
		for (int y = 1; y < plateDims[0].length - 1; y++) {

			int yIndex0 = y * plateDims[1].length;
			int yIndex1 = yIndex0 - plateDims[1].length;
			int yIndex2 = yIndex0 + plateDims[1].length;

			for (int x = 1; x < plateDims[1].length - 1; x++) {

					int index0 = yIndex0 + x;
					int index1 = yIndex0 + (x - 1);
					int index2 = yIndex0 + (x + 1);
					int index3 = yIndex1 + x;
					int index4 = yIndex2 + x;
					newPlate[index0] = 0.25 * (oldPlate[index1] +
							oldPlate[index2] +
							oldPlate[index3] +
							oldPlate[index4]);
			}
		}
	}

	// compare the computed final state with the state got from file
	bool valid = true;
	int mismatchCount = 0;
	for (int i = 0; i < plateSize; i++) {
		if (newPlate[i] - plateOut[i] > 0.1 || newPlate[i] - plateOut[i] < -0.1) {
			cout << "Mismatch found: computed value " << newPlate[i];
			cout << " read value " << plateOut[i] << "\n";
			valid = false;
			mismatchCount++;
		}
	}
	if (valid == true) {
		cout << "validation successful\n";
	} else {
		cout << "outputs mismatch in " << mismatchCount << " data points\n";
	}

	return 0;
}
