/*-----------------------------------------------------------------------------------
header file for the coordinator program
------------------------------------------------------------------------------------*/

#include "coordinator.h"

/*-----------------------------------------------------------------------------------
header files included for different purposes
------------------------------------------------------------------------------------*/

// for error reporting and diagnostics
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

// for math functions
#include <math.h>
#include <algorithm>

// for tuple definitions found in the source code
#include "tuple.h"
#include <vector>

// for LPU and PPU management data structures
#include "../codegen/structure.h"
#include "../codegen/lpu_management.h"

// for utility routines
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/common_utils.h"

// for routines related to partition functions
#include "../partition-lib/index_xform.h"
#include "../partition-lib/partition_mgmt.h"

// to input-output and initialization
#include "../runtime/input_prompt.h"
#include "../runtime/output_prompt.h"
#include "../runtime/allocator.h"

// for threading
#include <pthread.h>

// for synchronization
#include "../runtime/sync.h"

/*-----------------------------------------------------------------------------------
header files for tasks
------------------------------------------------------------------------------------*/

#include "initiate_lu.h"
#include "transposed_lu_factorization.h"
#include "subtract_matrix_multiply_result.h"

/*-----------------------------------------------------------------------------------
function for initializing program arguments
------------------------------------------------------------------------------------*/

ProgramArgs getProgramArgs() {
	ProgramArgs programArgs = ProgramArgs();
	programArgs.argument_matrix_file = inprompt::readString("argument_matrix_file");
	programArgs.block_size = inprompt::readPrimitive <int> ("block_size");
	programArgs.k = inprompt::readPrimitive <int> ("k");
	programArgs.l = inprompt::readPrimitive <int> ("l");
	programArgs.lower_matrix_file = inprompt::readString("lower_matrix_file");
	programArgs.pivot_matrix_file = inprompt::readString("pivot_matrix_file");
	programArgs.q = inprompt::readPrimitive <int> ("q");
	programArgs.upper_matrix_file = inprompt::readString("upper_matrix_file");
	return programArgs;
}

/*-----------------------------------------------------------------------------------
main function
------------------------------------------------------------------------------------*/

int main() {

	// starting execution timer clock
	struct timeval start;
	gettimeofday(&start, NULL);

	// creating a program log file
	std::cout << "Creating diagnostic log: it-program.log\n";
	std::ofstream logFile;
	logFile.open("it-program.log");

	// getting command line inputs
	ProgramArgs args = getProgramArgs();

	// declaring local variables
	int blockSize;
	ILUEnvironment *initEnv;
	int lastRow;
	TLUFEnvironment *luEnv;
	SMMREnvironment *mMultEnv;
	int max1;
	int max2;
	Range range;
	Range rows;

	//------------------------------------------ Coordinator Program

	blockSize = args.block_size;
	initEnv = new ILUEnvironment();
	initEnv->name = "Initiate LU";

	// substitute to load array
	Dimension aDims[2];
	inprompt::readArrayDimensionInfo("a", 2, aDims);
        initEnv->a = allocate::allocateArray <double> (2, aDims);
        allocate::randomFillPrimitiveArray <double> (initEnv->a, 2, aDims);
	initEnv->aDims[0].partition = aDims[0];
	initEnv->aDims[0].storage = aDims[0];
	initEnv->aDims[1].partition = aDims[1];
	initEnv->aDims[1].storage = aDims[1];

	// substitute to task invocation
	ILUPartition iluPartition;
	ilu::execute(initEnv, iluPartition, logFile);

	luEnv = new TLUFEnvironment();
	luEnv->name = "Transposed LU Factorization";

	// substitute to copying from one environment to another
	luEnv->u = initEnv->u;
	luEnv->uDims[0] = initEnv->uDims[0];
	luEnv->uDims[1] = initEnv->uDims[1];
	luEnv->l = initEnv->l;
	luEnv->lDims[0] = initEnv->lDims[0];
	luEnv->lDims[1] = initEnv->lDims[1];
	luEnv->p = NULL;

	rows = initEnv->aDims[0].partition.range;
	max1 = rows.max;
	max2 = initEnv->aDims[1].partition.range.max;
	{ // scope entrance for sequential loop
	int k;
	int iterationStart = rows.min;
	int iterationBound = rows.max;
	int indexIncrement = blockSize;
	int indexMultiplier = 1;
	if (rows.min > rows.max) {
		iterationBound *= -1;
		indexIncrement *= -1;
		indexMultiplier = -1;
	}
	for (k = iterationStart; 
			indexMultiplier * k <= iterationBound; 
			k += indexIncrement) {
		lastRow = k + blockSize - 1;
		if (lastRow > max1) {
			lastRow = max1;
		}
		range = Range();
		range.min = k;
		range.max = lastRow;

		// substitute to task invocation
		TLUFPartition luPartition;
		tluf::execute(luEnv, range, luPartition, logFile);

		if (lastRow < max1) {
			mMultEnv = new SMMREnvironment();
			mMultEnv->name = "Subtract Matrix Multiply Result";

			// substitute to copy from one environment to another
			mMultEnv->a = luEnv->u;
			mMultEnv->aDims[0] = luEnv->uDims[0].getSubrange(lastRow + 1, max1);
			mMultEnv->aDims[1] = luEnv->uDims[1].getSubrange(k, lastRow);
			mMultEnv->b = luEnv->l;
			mMultEnv->bDims[0] = luEnv->lDims[0].getSubrange(k, lastRow);
			mMultEnv->bDims[1] = luEnv->lDims[1].getSubrange(lastRow + 1, max2);
			mMultEnv->c = luEnv->u;
			mMultEnv->cDims[0] = luEnv->uDims[0].getSubrange(lastRow + 1, max1);
			mMultEnv->cDims[1] = luEnv->uDims[1].getSubrange(lastRow + 1, max2);
			
			// substitute to task invocation
			SMMRPartition smmrPartition;
			smmrPartition.k = args.k;
			smmrPartition.l = args.l;
			smmrPartition.q = args.q;
			smmr::execute(mMultEnv, smmrPartition, logFile);
		}
	}
	} // scope exit for sequential loop
	
	// calculating task running time
	struct timeval end;
	gettimeofday(&end, NULL);
	double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	logFile << "Execution Time: " << runningTime << " Seconds" << std::endl;
	logFile.close();

	// substitute to store array
	std::cout << "writing results to output files\n";
        if (outprompt::getYesNoAnswer("Want to save array \"a\" in a file?")) {
		Dimension aDims[2];
		aDims[0] = initEnv->aDims[0].storage;
		aDims[1] = initEnv->aDims[1].storage;
                outprompt::writeArrayToFile <double> ("a", initEnv->a, 2, aDims);
	}
        if (outprompt::getYesNoAnswer("Want to save array \"u\" in a file?")) {
		Dimension uDims[2];
		uDims[0] = luEnv->uDims[0].storage;
		uDims[1] = luEnv->uDims[1].storage;
                outprompt::writeArrayToFile <double> ("u", luEnv->u, 2, uDims);
	}
        if (outprompt::getYesNoAnswer("Want to save array \"l\" in a file?")) {
		Dimension lDims[2];
		lDims[0] = luEnv->lDims[0].storage;
		lDims[1] = luEnv->lDims[1].storage;
                outprompt::writeArrayToFile <double> ("l", luEnv->l, 2, lDims);
	}
        if (outprompt::getYesNoAnswer("Want to save array \"p\" in a file?")) {
		Dimension pDims[1];
		pDims[0] = luEnv->pDims[0].storage;
                outprompt::writeArrayToFile <int> ("p", luEnv->p, 1, pDims);
	}

	//--------------------------------------------------------------

	std::cout << "Parallel Execution Time: " << runningTime << " Seconds" << std::endl;
	return 0;
}
