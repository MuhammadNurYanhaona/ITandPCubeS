#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include "utils.h"
#include "structures.h"
#include "fileUtility.h"

int mainDPV() {
	Dimension uDims[1];
	double *u = readArrayFromFile <double> ("u", 1, uDims);
	Dimension vDims[1];
	double *v = readArrayFromFile <double> ("v", 1, vDims);
	double product = 0;
	for (int i = 0; i < uDims[0].length; i++) {
		product += u[i] * v[i];
	}
	std::cout << "Product:" << product << std::endl;
	return 0;
}



