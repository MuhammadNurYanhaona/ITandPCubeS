#include "list.h"
#include "structures.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include "utils.h"

using namespace std;

int mainS(int argc, char *argv[]) {

	char *fileName = NULL;
	if (argc > 1) {
		fileName = argv[1];
	} else {
		cout << "input file must be passed as a parameter\n";
		exit(EXIT_FAILURE);
	}
	int groupSize = 5;
	if (argc > 2) {
		groupSize = atoi(argv[2]);
	}

	List<double> *resultList = new List<double>;
	std::ifstream file(fileName);
	if (!file.is_open()) {
		cout << "could not open input file: " << fileName << "\n";
		exit(EXIT_FAILURE);
	}

	double total = 0.0;
	int iteration = 0;
	string line;
	while (getline(file, line)) {
		total += atof(line.c_str());
		iteration++;
		if (iteration == groupSize) {
			iteration = 0;
			double avg = total / groupSize;
			total = 0.0;
			int i = 0;
			for (i = 0 ; i < resultList->NumElements(); i++) {
				if (resultList->Nth(i) > avg) break;
			}
			resultList->InsertAt(avg, i);
		}
	}

	for (int i = 0; i < resultList->NumElements(); i++) {
		cout << resultList->Nth(i) << "\n";
	}

	return 0;
}



