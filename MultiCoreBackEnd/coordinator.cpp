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
	{ // scope starts for load-array operation
	Dimension arrayDims[2];
	if (outprompt::getYesNoAnswer("Want to read array \"initEnv->a\" from a file?")) {
		initEnv->a = inprompt::readArrayFromFile <double> ("initEnv->a", 
			2, arrayDims, args.argument_matrix_file);
	} else {
		inprompt::readArrayDimensionInfo("initEnv->a", 2, arrayDims);
		initEnv->a = allocate::allocateArray <double> (2, arrayDims);
		allocate::randomFillPrimitiveArray <double> (initEnv->a, 
			2, arrayDims);
	}
	initEnv->aDims[0].partition = arrayDims[0];
	initEnv->aDims[0].storage = arrayDims[0].getNormalizedDimension();
	initEnv->aDims[1].partition = arrayDims[1];
	initEnv->aDims[1].storage = arrayDims[1].getNormalizedDimension();
	} // scope ends for load-array operation
	{ // scope starts for invoking: Initiate LU
	ILUPartition partition;
	ilu::execute(initEnv, partition, logFile);
	} // scope ends for task invocation
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
		{ // scope starts for invoking: Transposed LU Factorization
		TLUFPartition partition;
		tluf::execute(luEnv, range, partition, logFile);
		} // scope ends for task invocation
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

			{ // scope starts for invoking: Subtract Matrix Multiply Result
			SMMRPartition partition;
			partition.k = args.k;
			partition.l = args.l;
			partition.q = args.q;
			smmr::execute(mMultEnv, partition, logFile);
			} // scope ends for task invocation
		}
	}
	} // scope exit for sequential loop
	if (outprompt::getYesNoAnswer("Want to save array \"initEnv->a\" in a file?")) {
		Dimension arrayDims[2];
		arrayDims[0] = initEnv->aDims[0].storage;
		arrayDims[1] = initEnv->aDims[1].storage;
		outprompt::writeArrayToFile <double> ("initEnv->a", 
			initEnv->a, 2, arrayDims, args.argument_matrix_file);
	}
	if (outprompt::getYesNoAnswer("Want to save array \"luEnv->u\" in a file?")) {
		Dimension arrayDims[2];
		arrayDims[0] = luEnv->uDims[0].storage;
		arrayDims[1] = luEnv->uDims[1].storage;
		outprompt::writeArrayToFile <double> ("luEnv->u", 
			luEnv->u, 2, arrayDims, args.upper_matrix_file);
	}
	if (outprompt::getYesNoAnswer("Want to save array \"luEnv->l\" in a file?")) {
		Dimension arrayDims[2];
		arrayDims[0] = luEnv->lDims[0].storage;
		arrayDims[1] = luEnv->lDims[1].storage;
		outprompt::writeArrayToFile <double> ("luEnv->l", 
			luEnv->l, 2, arrayDims, args.lower_matrix_file);
	}
	if (outprompt::getYesNoAnswer("Want to save array \"luEnv->p\" in a file?")) {
		Dimension arrayDims[1];
		arrayDims[0] = luEnv->pDims[0].storage;
		outprompt::writeArrayToFile <int> ("luEnv->p", 
			luEnv->p, 1, arrayDims, args.pivot_matrix_file);
	}

	//--------------------------------------------------------------

	// calculating task running time
	struct timeval end;
	gettimeofday(&end, NULL);
	double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	logFile << "Execution Time: " << runningTime << " Seconds" << std::endl;

	logFile.close();
	std::cout << "Parallel Execution Time: " << runningTime << " Seconds" << std::endl;
	return 0;
}
