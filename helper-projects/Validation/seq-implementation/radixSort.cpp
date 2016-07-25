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
#include <cmath>

#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"

using namespace std;

int mainRSort() {

	//----------------------------------------------------------------------------- initialize the program

	// read the unordered input array from a file
	Dimension keyDim[1];
	int *keysIn = readArrayFromFile <int> ("input array of keys", 1, keyDim);

	// specify the number of binary digits in the largest key
	cout << "Enter the number of binary digits required to express the largest key\n";
	int keySize;
	cin >> keySize;

	// specify the number of binary digits covered by each round of distribution sort
	cout << "Enter the number of binary digit places each round of distribution sort should process\n";
	int digits;
	cin >> digits;

	// declare two output arrays of keys to store the intermediate results of key sorting
	int keyCount = keyDim[0].length;
	int *keysOut0 = new int[keyCount];
	int *keysOut1 = new int[keyCount];

	// initialize the output arrays with the content of the input file
	for (int i = 0; i < keyDim[0].length; i++) {
		keysOut0[i] = keysOut1[i] = keysIn[i];
	}

	// declare an scatter offsets tracking array
	int *scatter_offsets = new int[keyCount];

	//--------------------------------------------------------------------------- start timing calculation

	// starting execution timer clock
	struct timeval start;
	gettimeofday(&start, NULL);

	//--------------------------------------------------------------------------------- perform radix sort

	// initialize section---------------------------------------------------------------------------------
	int radix = pow(digits, 2);
	int sortingRounds = keySize / digits;
	int *g0 = new int[radix];
	int *g1 = new int[radix];

	// start computation----------------------------------------------------------------------------------

	// perform sorting-rounds number of key distribution sorts
	int *keys_next, *keys_curr;
	for (int round = 0; round < sortingRounds; round++) {

		// determine the input and output arrays for the current sorting round
		if (round % 2 == 0) {
			keys_next = keysOut1;
			keys_curr = keysOut0;
		} else {
			keys_next = keysOut0;
			keys_curr = keysOut1;
		}

		// up-sweep reduction step-----------------------------------------------------------------------
		// determine the starting index of keys having different radix values in the current digit places
		// of interest
		for (int i = 0; i < radix; i++) g0[i] = 0;
		int shiftCount = digits * round;
		for (int i = 0; i < keyCount; i++) {

			// determine the radix value of the current key in the digit place of interest
			int currentKey = keys_curr[i];
			int radixValue = (currentKey >> shiftCount) & (radix - 1);

			// then increment the counter tracking the number of keys having the same value for the digit
			g0[radixValue]++;
		}

		// top-level scan step---------------------------------------------------------------------------
		g1[0] = 0;
		for (int i = 1; i < radix; i++) {
			g1[i] = g0[i - 1] + g1[i - 1];
		}

		// scatter offset step---------------------------------------------------------------------------
		for (int i = 0; i < keyCount; i++) {
			int currentKey = keys_curr[i];
			int radixValue = (currentKey >> shiftCount) & (radix - 1);
			scatter_offsets[i] = g1[radixValue];
			g1[radixValue]++;
		}

		// rearrange the keys step-----------------------------------------------------------------------
		for (int i = 0; i < keyCount; i++) {
			keys_next[scatter_offsets[i]] = keys_curr[i];
		}
	}

	//---------------------------------------------------------------------------- calculate running time

	struct timeval end;
	gettimeofday(&end, NULL);
	double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;

	return 0;
}
