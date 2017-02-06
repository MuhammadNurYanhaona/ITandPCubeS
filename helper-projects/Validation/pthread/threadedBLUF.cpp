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
#include <pthread.h>

#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"

using namespace std;

//---------------------------------------------------------------------------- program data

double *a;
Dimension aDims[2];
double *u;
Dimension uDims[2];
double *l;
Dimension lDims[2];
int *p;
Dimension pDims[1];

//----------------------------------------------------------------- Thread Interaction data

int threadCount;
int blockSize;
pthread_barrier_t barrier;
int pivot;
double *p_column;

//------------------------------------------------------------------------- Thread Function

void *computeBLUF(void *arg) {

	int threadId = *((int*) arg);
	int blockStride = blockSize * threadCount;
	
	// ******************************************************** Initialize LU task code
	for (int i = 0; i < aDims[0].length; i++) {
		int aRow = i * aDims[1].length;
		for (int j = blockSize * threadId; j < aDims[1].length; j += blockStride) {
			int start = j;
			int end = j + blockSize - 1;
			if (end >= aDims[1].length) end = aDims[1].length - 1;
			for (int r = start; r <= end ; r++) {
				int uRow = r * uDims[1].length;
				u[uRow + i] = a[aRow + r];
			}
		}
	}
	for (int i = blockSize * threadId; i < lDims[0].length; i += blockStride) {
		int start = i;
		int end = i + blockSize - 1;
		if (end >= lDims[0].length) end = lDims[0].length - 1;
		for (int c = start; c <= end; c++) {
			int cols = lDims[1].length;
			l[c * cols + c] = 1;
		}
	}
	
	// join pthread barrier to ensure initialization of U and L is done
	pthread_barrier_wait(&barrier);

	Range rows = aDims[0].range;
	int cols = uDims[1].length;
        int max1 = rows.max;
        int max2 = aDims[1].range.max;

        for (int k = 0; k <= rows.max; k += blockSize) {
                int lastRow = k + blockSize - 1;
                if (lastRow > max1) lastRow = max1;
                Range range;
                range.min = k;
                range.max = lastRow;

		// execute code equivalent to Transposed LU Factorization task
                {
                        // repeat loop
                        for (int k = range.min; k <= range.max; k++) {

                                //**************************************** Select Pivot step
				
				// determine which thread will do the pivot selection
				int blockNo = k / blockStride;
				int blockIndex = k % blockStride;
				int strideIndex = blockIndex / blockSize; 				
				
				// do the pivot selection in the selected thread
				if (strideIndex == threadId) {
					double max = u[k * cols + k];
					pivot = k;
					int kRow = k * cols;
					for (int j = k; j < uDims[1].length; j++) {
						if (u[kRow + j] > max) {
							max = u[kRow + j];
							pivot = j;
						}
					}
				
					// store the pivot
					p[k] = pivot;

				}

				// wait on the barrier to make the pivot available to all
				pthread_barrier_wait(&barrier);

				//************************************ Interchange Rows step
				
				if (k != pivot) {

					// determine the proper starting index for a thread
					int i = blockNo * blockStride + blockSize * threadId;
					// advance a stride if the thread's block ends before reaching k
					if (i + blockSize - 1 < k) i += blockStride;

					// update U
                                        for (; i < uDims[0].length; i += blockStride) {
						int start = i;
						if (start < k) start = k;
						int end = i + blockSize - 1;
						if (end >= uDims[0].length) end = uDims[0].length - 1;
						for (int r = start; r <= end; r++) {
                                                	double pivotEntry = u[r * cols + k];
                                                	u[r * cols + k] = u[r * cols + pivot];
                                                	u[r * cols + pivot] = pivotEntry;
						}
                                        }

					// update L
                                        for (i = blockSize * threadId; i < k; i += blockStride) {
						int start = i;
						int end = i + blockSize - 1;
						if (end >= k) end = k - 1;
						for (int r = start; r <= end; r++) {
                                                	double pivotEntry = l[r * cols + k];
                                                	l[r * cols + k] = l[r * cols + pivot];
                                                	l[r * cols + pivot] = pivotEntry;
						}
                                        }
                                }

				//*************************************** Update Lower step
			
				// the same thread that did the pivot selection is responsible for the
				// recording of next column of L (L is transposed)	
				if (strideIndex == threadId) {
					for (int j = k + 1; j < lDims[1].length; j++) {
                                        	l[k * cols + j] = u[k * cols + j] / u[k * cols + k];
                                        	u[k * cols + j] = 0;
                                	}
				}
				
				// wait on barrier to make the change of L visible to all threads
				pthread_barrier_wait(&barrier);
				
				//***************************** Update Upper Row Block step 
	
				// determine the starting index for update for a thread
				int i = blockNo * blockStride + blockSize * threadId;
				// advance a stride if the thread's block ends before reaching k + 1
				if (i + blockSize - 1 < k + 1) i += blockStride;

				for (; i <= range.max; i += blockStride) {
					int start = i;
					if (start <= k) start = k + 1;
					int end = i + blockSize - 1;
					if (end > range.max) end = range.max;
					for (int r = start; r <= end; r++) {	
						for (int j = k + 1; j < uDims[1].length; j++) {
							u[r * cols + j] -= l[k * cols + j] * u[r * cols + k];
						}
					}
				}
				
				//****************************** Generate Pivot Column step
				
				// determine the starting index for update for a thread
				blockNo = range.min / blockStride;
				i = blockNo * blockStride + blockSize * threadId;
				// advance a stride if the thread's block ends before reaching range.min
				if (i + blockSize - 1 < range.min) i += blockStride;

				for (; i < k; i += blockStride) {
					int start = i;
					if (start < range.min) start = range.min;	
					int end = i + blockSize - 1;
					if (end >= k) end = k - 1;
					for (int r = start; r <= end; r++) {	
						p_column[r] = l[r * cols + k];
					}
				}

				// wait on the barrier to make available the new p_column
				pthread_barrier_wait(&barrier);
				
				//******************************** Update Upper Column step

				// determine the starting index for update for a thread
				blockNo = (range.max + 1) / blockStride;
				i = blockNo * blockStride + blockSize * threadId;
				// advance a stride if the thread's block ends before reaching range.max + 1
				if (i + blockSize < range.max + 1) i += blockStride;

				for (; i < uDims[0].length; i += blockStride) {
					int start = i;
					if (start <= range.max) start = range.max + 1;
					int end = i + blockSize - 1;
					if (end >= uDims[0].length) end = uDims[0].length - 1;
					for (int r = start; r <= end; r++) { 
						for (int j = range.min; j < k; j++) {
							u[r * cols + k] -= u[r * cols + j] * p_column[j];
						}
					}
				}

				// wait on the barrier to sink everyone for the next iteration
				pthread_barrier_wait(&barrier);
			
			} // end of repeat loop
		} // end of the Transposed LU Factorization sub-part code

		if (lastRow < max1) { // conditional execution of the SAXPY task

			//****************************************************** SAXPY step

			int cols = uDims[1].length;
			int saxpyBlockId = 0;	// a variable used to evenly distribute work among threads
			for (int iB = lastRow + 1; iB <= max1; iB += blockSize) {
				int rStart = iB;
				int rEnd = rStart + blockSize - 1;
				if (rEnd > max1) rEnd = max1;
				for (int jB = lastRow + 1; jB <= max2; jB += blockSize) {

					// We increase the ID counter first then do the decision making if
					// the current thread is going to do the work. This is done to make
					// sure every thread goes through the same ID sequence comparison,
					// and the SAXPY work is uniformly distributed.
					saxpyBlockId++;
					if ((saxpyBlockId - 1) % threadCount != threadId) continue;

					int cStart = jB;
					int cEnd = cStart + blockSize - 1;
					if (cEnd > max2) cEnd = max2;
					for (int kB = k; kB <= lastRow; kB += blockSize) {

						// Work distribution is not done in this step as this would
						// result in different threads trying to update the same
						// entries in the left-hand-side.

						int startIndex = kB;
						int endIndex = startIndex + blockSize - 1;
						if (endIndex > lastRow) endIndex = lastRow;
						for (int i = rStart; i <= rEnd; i++) {
							for (int j = cStart; j <= cEnd; j++) {
								double total = 0.0;
								for (int c = startIndex; c <= endIndex; c++) {
									total += u[i * cols + c] * l[c * cols + j];
								}
								u[i * cols + j] -= total;
							}
						} // ending blocked computation
					} // ending of blocking of common dimension
				} // ending of blocking of B columns
			} // ending of blocking of A rows

			// join the barrier to ensure everyone has up-to-date data for the next LUF step
			pthread_barrier_wait(&barrier);

		} // end of the SAXPY step
	}	
}

//--------------------------------------------------------------------------- Main Function

void initializeARandomly(int dimLength) {

	aDims[0].range.min = 0;
	aDims[0].range.max = dimLength - 1;
	aDims[0].length = dimLength;
	aDims[1] = aDims[0];

	int elementCount = dimLength * dimLength;
	a = new double[elementCount];

	srand (time(NULL));
	for (int i = 0; i < elementCount; i++) {
		a[i] = ((rand() % 100) / 75.00f);
	}
}

int mainTBLUF(int argc, char *argv[]) {

	if (argc < 3) {
		cout << "There are two modes for using this program\n";
		cout << "If you want to read/write data from/to files then pass two arguments\n";
		cout << "\t1. Provide the block size for partitioning the argument matrix\n";
		cout << "\t2. Provide the number of threads\n";
		cout << "If you rather want to check timing on randomly generated data then pass 1 more argument\n";
		cout << "\t3. The dimension length of the square argument matrix\n";
		exit(EXIT_FAILURE);		
	}

	// parse command line arguments
	bool fileIOMode = true;
	blockSize = atoi(argv[1]);	
	threadCount = atoi(argv[2]);
	int dimensionLength;
	if (argc >= 4) {
		fileIOMode = false;
		dimensionLength = atoi(argv[3]);	
		
	}

	// read argument matrix from file
	struct timeval start;
	gettimeofday(&start, NULL);
	if (fileIOMode) {
		a = readArrayFromFile <double> ("a", 2, aDims);
	} else {
		initializeARandomly(dimensionLength);
	}
	struct timeval end;
	gettimeofday(&end, NULL);
	double readingTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	cout << "Data reading/intialization time: " << readingTime << " Seconds\n";

	// initialize necessary data structures
	gettimeofday(&start, NULL);
	pDims[0] = uDims[0] = lDims[0] = aDims[0];
	uDims[1] = lDims[1] = aDims[1];
	int uSize = uDims[0].length * uDims[1].length;
	u = new double[uSize];
	for (int i = 0; i < uSize; i++) u[i] = 0;
	int lSize = lDims[0].length * lDims[1].length;
	l = new double[lSize];
	for (int i = 0; i < lSize; i++) l[i] = 0;
	p = new int[pDims[0].length];
	for (int i = 0; i < pDims[0].length; i++) p[i] = 0;
	p_column = new double[uDims[0].length];
	for (int i = 0; i < uDims[0].length; i++) p_column[i] = 0;

	// initialize the pthread_barrier
	pthread_barrier_init(&barrier, NULL, threadCount);
	gettimeofday(&end, NULL);
	double setupTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	cout << "data structure setup overhead: " << setupTime << " Seconds\n";

	// start the threads
	gettimeofday(&start, NULL);
	// create arrays of 5 thread Ids and thread objects
        int threadIds[threadCount];
        pthread_t threads[threadCount];
	for (int i = 0; i < threadCount; i++) {
                threadIds[i] = i;
                int status = pthread_create(&threads[i], NULL, computeBLUF, (void*) &threadIds[i]);
                if (status != 0) {
                        cout << "Could not create some pthreads\n";
                        exit(EXIT_FAILURE);
                }
        }

	// join threads
	for (int i = 0; i < threadCount; i++) {
                pthread_join(threads[i], NULL);
        }

	gettimeofday(&end, NULL);
	double computationTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	cout << "computation time: " << computationTime << " Seconds\n";

	// write results
	gettimeofday(&start, NULL);
	if (fileIOMode) {
		writeArrayToFile<double>("u", u, 2, uDims);
		writeArrayToFile<double>("l", l, 2, lDims);
		writeArrayToFile<int>("p", p, 1, pDims);
	}
	gettimeofday(&end, NULL);
	double writingTime = ((end.tv_sec + end.tv_usec / 1000000.0)
			- (start.tv_sec + start.tv_usec / 1000000.0));
	cout << "data writing time if applicable: " << writingTime << " Seconds\n";

	// report running time
	cout << "total time: " << readingTime + setupTime + computationTime + writingTime << "\n";

	return 0;
}



