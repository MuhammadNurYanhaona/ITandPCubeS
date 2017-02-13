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
#include <mpi.h>

#include "../utils.h"
#include "../structures.h"
#include "../fileUtility.h"
#include "../stream.h"

using namespace std;

//-------------------------------------------------------------- Data Dimension Information

Dimension aDims[2];
Dimension uDims[2];
Dimension lDims[2];
Dimension pDims[1];
Dimension uBlockDims[2];
Dimension lBlockDims[2];

//--------------------------------------------------------------- Data Structure References

double *a;
double *u;
double *l;
double *p;
double *uBlock;
double *lBlock;
double *lRow;
double *pColumn;
int pivot;

//-------------------------------------------------------------- Partition Config Variables

int blockSize;
int rank;
int processCount;


//-------------------------------------------------------------------- Supporting functions

int getPartLength(Dimension dimension) {

	int stride = blockSize * processCount;
        int strideCount = dimension.length / stride;
        int partialStrideElements = dimension.length % stride;
        int blockCount = partialStrideElements / blockSize;
        int extraEntriesBefore = partialStrideElements;
        int myEntries = strideCount * blockSize;

	// if extra entries fill up a complete new block in the stride of the current LPU then
        // its number of entries should increase by the size parameter and extra preceeding
        // entries should equal to size * preceeding stride count
        if (blockCount > rank) {
                myEntries += blockSize;
        // If the extra entries does not fill a complete block for the current one then it should
        // have whatever remains after filling up preceeding blocks
        } else if (blockCount == rank) {
                myEntries += partialStrideElements - blockCount * blockSize;
        }

	return myEntries;
}

int getStorageIndex(int index) {
	int stride = blockSize * processCount;
	int startOfBlock = blockSize * rank;
	int relativeIndex = index - startOfBlock;
	int strideId = relativeIndex / stride;
	int strideIndex = relativeIndex % blockSize;
	return strideId * blockSize + strideIndex;	
}

void initializeMetadata() {
	pDims[0] = uDims[0] = lDims[0] = aDims[0];
        uDims[1] = lDims[1] = aDims[1];
        uBlockDims[0] = uDims[0];
        uBlockDims[1].range.min = 0;
        uBlockDims[1].range.max = blockSize - 1;
        uBlockDims[1].length = blockSize;
        lBlockDims[1] = lDims[0];
        lBlockDims[0] = uBlockDims[1];
}

void allocateArrays() {
	int aLength = aDims[0].length * getPartLength(aDims[1]);
	a = new double[aLength];
	int uLength = getPartLength(uDims[0]) * uDims[1].length;
	u = new double[uLength];
	l = new double[uLength];
	for (int i = 0; i < uLength; i++) l[i] = 0.0;
	int lBlockLength = lBlockDims[0].length * lBlockDims[1].length;
	lBlock = new double[lBlockLength];
	if (rank == 0) {
		p = new double[pDims[0].length];
	} else p = NULL;
	lRow = new double[lDims[1].length];
	pColumn = new double[blockSize];
}

void initializeARandomly() {
	int aLength = aDims[0].length * getPartLength(aDims[1]);
        srand (time(NULL));
        for (int i = 0; i < aLength; i++) {
                a[i] = ((rand() % 100) / 75.00f);
        }
}

void readAFromFile(const char *filePath) {

	TypedInputStream<double> *stream = new TypedInputStream<double>(filePath);
	int storeIndex = 0;
        int blockStride = blockSize * processCount;
	List<int> *indexList = new List<int>;

	stream->open();
        for (int i = 0; i < aDims[0].length; i++) {
                for (int j = blockSize * rank; j < aDims[1].length; j += blockStride) {
			int start = j;
			int end = start + blockSize - 1;
			if (end >= aDims[1].length) end = aDims[1].length - 1;
			for (int c = start; c <= end; c++) {
				indexList->clear();
				indexList->Append(i);
				indexList->Append(c);
				a[storeIndex] = stream->readElement(indexList);
				storeIndex++;
			}
		}
	}
	stream->close();

	delete indexList;
	delete stream;
}

void writeAToFile() {

	// wait for your turn
	if (rank != 0) {
		int predecessorDone = 0;
                MPI_Status status;
                MPI_Recv(&predecessorDone, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);	
	}

	List<Dimension*> *dimLengths = new List<Dimension*>;
	dimLengths->Append(&aDims[0]);
	dimLengths->Append(&aDims[1]);
	TypedOutputStream<double> *stream = new TypedOutputStream<double>("/home/yan/a-copy.bin", dimLengths, rank == 0);	
	int storeIndex = 0;
	List<int> *indexList = new List<int>;
        int blockStride = blockSize * processCount;
	
	stream->open();
        for (int i = 0; i < aDims[0].length; i++) {
                int aRow = i * aDims[1].length;
                for (int j = blockSize * rank; j < aDims[1].length; j += blockStride) {
			int start = j;
			int end = start + blockSize - 1;
			if (end >= aDims[1].length) end = aDims[1].length - 1;
			for (int c = start; c <= end; c++) {
				indexList->clear();
				indexList->Append(i);
				indexList->Append(c);
				stream->writeElement(a[storeIndex], indexList);
				storeIndex++;
			}
		}
	}
	stream->close();
	delete indexList;
	delete stream;

	// notify the next in line
	if (rank < processCount - 1) {
                int writingDone = 1;
                MPI_Send(&writingDone, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
	}
}

void writeUandLToFile() {
	
	// wait for your turn
	if (rank != 0) {
		int predecessorDone = 0;
                MPI_Status status;
                MPI_Recv(&predecessorDone, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);	
	}
	
	List<Dimension*> *udimLengths = new List<Dimension*>;
	udimLengths->Append(&uDims[0]);
	udimLengths->Append(&uDims[1]);
	TypedOutputStream<double> *ustream = new TypedOutputStream<double>("/home/yan/u.bin", udimLengths, rank == 0);	
	List<Dimension*> *ldimLengths = new List<Dimension*>;
	ldimLengths->Append(&lDims[0]);
	ldimLengths->Append(&lDims[1]);
	TypedOutputStream<double> *lstream = new TypedOutputStream<double>("/home/yan/l.bin", ldimLengths, rank == 0);	
	
	List<int> *indexList = new List<int>;
        int blockStride = blockSize * processCount;
	int storeIndex = 0;

	ustream->open();
	lstream->open();
        for (int i = blockSize * rank; i < aDims[0].length; i+= blockStride) {
		int start = i;
		int end = start + blockSize - 1;
		if (end >= aDims[0].length) end = aDims[0].length - 1;
		for (int r = start; r <= end; r++) {
			for (int j = 0; j < aDims[1].length; j++) {
				indexList->clear();
				indexList->Append(r);
				indexList->Append(j);
				ustream->writeElement(u[storeIndex], indexList);
				lstream->writeElement(l[storeIndex], indexList);
				storeIndex++;
			}
		}
	}
	ustream->close();
	lstream->close();

	delete indexList;
	delete ustream;
	delete lstream;

	// notify the next in line
	if (rank < processCount - 1) {
                int writingDone = 1;
                MPI_Send(&writingDone, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
	}
}

void writePToFile() {
	List<Dimension*> *dimLengths = new List<Dimension*>;
	dimLengths->Append(&pDims[0]);
	TypedOutputStream<int> *stream = new TypedOutputStream<int>("/home/yan/p.bin", dimLengths, true);	
	stream->open();
	for (int i = 0; i < pDims[0].length; i++) {
		stream->writeNextElement(p[i]);
	}
	stream->close();
	delete stream;
}

//--------------------------------------------------------------------- Block LUF functions

void prepare() {
        int blockStride = blockSize * processCount;
	int aColumnCount = getPartLength(aDims[1]); 
	for (int i = 0; i < aDims[0].length; i++) {
                for (int j = blockSize * rank; j < aDims[1].length; j += blockStride) {
                        int start = j;
                        int end = j + blockSize - 1;
                        if (end >= aDims[1].length) end = aDims[1].length - 1;
                        for (int r = start; r <= end ; r++) {
				int aIndex = i * aColumnCount + getStorageIndex(r);
				int uIndex = getStorageIndex(r) * uDims[1].length + i;
                                u[uIndex] = a[aIndex];
                        }
                }
        }
        for (int i = blockSize * rank; i < lDims[0].length; i += blockStride) {
                int start = i;
                int end = i + blockSize - 1;
                if (end >= lDims[0].length) end = lDims[0].length - 1;
                for (int c = start; c <= end; c++) {
			int lIndex = getStorageIndex(c) * lDims[1].length + c;
                        l[lIndex] = 1;
                }
        }
}

void selectPivot(int k) {
	int kRow = getStorageIndex(k) * uDims[1].length;
	double max = u[kRow + k];
	pivot = k;
	for (int j = k; j < uDims[1].length; j++) {
		if (u[kRow + j] > max) {
			max = u[kRow + j];
			pivot = j;
		}
	}
}

void interchangeRows(int pivot, int k) {

	// determine the proper starting index for a process
        int blockStride = blockSize * processCount;
	int blockNo = k / blockStride;
	int cols = uDims[1].length;
	int i = blockNo * blockStride + blockSize * rank;
	// advance a stride if the process's block ends before reaching k
	if (i + blockSize - 1 < k) i += blockStride;

	// update U
	for (; i < uDims[0].length; i += blockStride) {
		int start = i;
		if (start < k) start = k;
		int end = i + blockSize - 1;
		if (end >= uDims[0].length) end = uDims[0].length - 1;
		for (int r = start; r <= end; r++) {
			int rowIndex = getStorageIndex(r) * cols;
			double pivotEntry = u[rowIndex + k];
			u[rowIndex + k] = u[rowIndex + pivot];
			u[rowIndex + pivot] = pivotEntry;
		}
	}

	// update L
	for (i = blockSize * rank; i < k; i += blockStride) {
		int start = i;
		int end = i + blockSize - 1;
		if (end >= k) end = k - 1;
		for (int r = start; r <= end; r++) {
			int rowIndex = getStorageIndex(r) * cols;
			double pivotEntry = l[rowIndex + k];
			l[rowIndex + k] = l[rowIndex + pivot];
			l[rowIndex + pivot] = pivotEntry;
		}
	}
}

void updateL(int k) {
	int cols = uDims[1].length;
	int kStore = getStorageIndex(k);
	for (int j = k + 1; j < lDims[1].length; j++) {
		l[kStore * cols + j] = u[kStore * cols + j] / u[kStore * cols + k];
		u[kStore * cols + j] = 0;
		lRow[j] = l[kStore * cols + j];
	}
}

void updateURowsBlock(int k, Range range) {
        
	// determine the starting index for update for a process
	int blockStride = blockSize * processCount;
	int blockNo = k / blockStride;
	int i = blockNo * blockStride + blockSize * rank;
	// advance a stride if the process's block ends before reaching k + 1
	if (i + blockSize - 1 < k + 1) i += blockStride;

	int cols = uDims[1].length;
	for (; i <= range.max; i += blockStride) {
		int start = i;
		if (start <= k) start = k + 1;
		int end = i + blockSize - 1;
		if (end > range.max) end = range.max;
		for (int r = start; r <= end; r++) {
			int rIndex = getStorageIndex(r) * cols;
			for (int j = k + 1; j < uDims[1].length; j++) {
				u[rIndex + j] -= lRow[j] * u[rIndex + k];
			}
		}
	}
}

void generatePivotColumn(int k, Range range) {
	int cols = lDims[1].length;
	for (int i = range.min; i < k; i++) {
		int pIndex = i - range.min;
		int lIndex = getStorageIndex(i) * cols + k;
		pColumn[pIndex] = l[lIndex]; 
	}
}

void updateUColsBlock(int k, Range range) {

	// determine the starting index for update for a process
	int blockStride = blockSize * processCount;
	int blockNo = (range.max + 1) / blockStride;
	int i = blockNo * blockStride + blockSize * rank;
	// advance a stride if the process's block ends before reaching range.max + 1
	if (i + blockSize < range.max + 1) i += blockStride;

	int cols = uDims[1].length;
	for (; i < uDims[0].length; i += blockStride) {
		int start = i;
		if (start <= range.max) start = range.max + 1;
		int end = i + blockSize - 1;
		if (end >= uDims[0].length) end = uDims[0].length - 1;
		for (int r = start; r <= end; r++) {
			int rowIndex = getStorageIndex(r) * cols;
			for (int j = range.min; j < k; j++) {
				u[rowIndex + k] -= u[rowIndex + j] * pColumn[j - range.min];
			}
		}
	}
}

void copyUpdatedLBlock(Range range) {
	for (int i = range.min; i <= range.max; i++) {
		int rowIndex1 = (i - range.min) * lBlockDims[1].length;
		int rowIndex2 = getStorageIndex(i) * lDims[1].length;
		for (int j = range.max + 1; j <= lDims[1].length; j++) {
			lBlock[rowIndex1 + j] = l[rowIndex2 + j];
		}
	}
}

void saxpy(Range range) {
	
        int cols = uDims[1].length;
        int max1 = uDims[0].range.max;
        int max2 = uDims[1].range.max;

	// determine the starting stride for the un-updated portion of U for a process
        int blockStride = blockSize * processCount;
	int blockNo = (range.max + 1) / blockStride;
	int iB = blockNo * blockStride + blockSize * rank;
	// advance a stride if the process's block ends before reaching range.max + 1
	if (iB + blockSize - 1 < range.max + 1) iB += blockStride;
	
	for (; iB <= max1; iB += blockStride) {
		int rStart = iB;
		if (rStart < range.max + 1) rStart = range.max + 1;
		int rEnd = iB + blockSize - 1;
		if (rEnd > max1) rEnd = max1;
		for (int jB = range.max + 1; jB <= max2; jB += blockSize) {
			int cStart = jB;
			int cEnd = cStart + blockSize - 1;
			if (cEnd > max2) cEnd = max2;
			for (int kB = range.min; kB <= range.max; kB += blockSize) {
				int startIndex = kB;
				int endIndex = startIndex + blockSize - 1;
				if (endIndex > range.max) endIndex = range.max;

				for (int i = rStart; i <= rEnd; i++) {
					int uRowIndex = getStorageIndex(i) * cols;
					for (int j = cStart; j <= cEnd; j++) {
						double total = 0.0;
						for (int c = startIndex; c <= endIndex; c++) {
							int lBlockRowIndex = (c - range.min) * cols;
							total += u[uRowIndex + c] * lBlock[lBlockRowIndex + j];
						}
						u[uRowIndex + j] -= total;
					}
				} // ending blocked computation

			} // ending of blocking of the common dimension
		} // ending of blocking of B columns
	} // ending of blocking of A rows

}

//--------------------------------------------------------------------------- Main Function

int main(int argc, char *argv[]) {
	
	// do MPI intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processCount);

	if (argc < 4) {
		if (rank == 0) {
			cout << "There are two modes for using this program\n";
			cout << "If you want to read/write data from/to files then pass arguments as follows\n";
			cout << "\t1. block size for partitioning\n";
			cout << "\t2. a 0 indicating file IO mode\n";
			cout << "\t3. path of the binary input file\n";
			cout << "If you rather want to check timing on randomly generated data then pass arguments as follows\n";
			cout << "\t1. block size for partitioning\n";
			cout << "\t2. a 1 indicating performance testing mode\n";
			cout << "\t3. The dimension length of the square argument matrix\n";
		}
		MPI_Finalize();
		exit(EXIT_FAILURE);		
	}

	// start timer
	struct timeval start;
        gettimeofday(&start, NULL);

	// parse command line arguments
	blockSize = atoi(argv[1]);
	bool fileIOMode (atoi(argv[2]) == 0);	
	int dimLength;
	if (!fileIOMode) {
		dimLength = atoi(argv[3]);
		aDims[0].range.min = 0;
		aDims[0].range.max = dimLength - 1;
		aDims[0].length = dimLength;
		aDims[1] = aDims[0];			
		initializeMetadata();
		allocateArrays();
		initializeARandomly();
	} else {
		const char *filePath = argv[3];
		std::ifstream file(filePath);
        	if (!file.is_open()) {
                	std::cout << "could not open the specified file\n";
                	std::exit(EXIT_FAILURE);
        	}
		readArrayDimensionInfoFromFile(file, 2, aDims);
		file.close();
		initializeMetadata();
		allocateArrays();
		readAFromFile(filePath);
	}

	//------------------------------------------------------------ Computation Starts 

	// execute prepare step that initialize U and L from A
	prepare();

	Range rows = aDims[0].range;
        int cols = uDims[1].length;
        int max1 = rows.max;
        int max2 = aDims[1].range.max;
        int blockStride = blockSize * processCount;

	// iterate over the rows of argument matrix in block-by-block basis
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

				// determine which process will do the pivot selection
                                int blockNo = k / blockStride;
                                int blockIndex = k % blockStride;
                                int strideIndex = blockIndex / blockSize;

				// do the pivot selection in the selected process
                                if (strideIndex == rank) {
					selectPivot(k);
				}

				// perform an MPI broadcast of the pivot variable
				MPI_Bcast(&pivot, 1, MPI_INT, strideIndex, MPI_COMM_WORLD);

				// store the pivot in the pivot array if the process rank is 0
				if (rank == 0) {
					p[k] = pivot;
				}

				// interchange rows if needed
				if (k != pivot) {
					interchangeRows(pivot, k);
				}

				// the same thread that did the pivot selection is responsible for the
                                // recording of next column of L (L is transposed)      
                                if (strideIndex == rank) {
					updateL(k);
                                }

				// perform an MPI broadcast of the updated row of l
				MPI_Bcast(lRow, cols, MPI_DOUBLE, strideIndex, MPI_COMM_WORLD);

				// perform the update of a selected sequence of rows of U
				updateURowsBlock(k, range);

				// generate the pivot column for updating remainder of U
                                if (strideIndex == rank) {
					generatePivotColumn(k, range);
				}

				// perform an MPI broadcast to share the pivot column
				MPI_Bcast(pColumn, blockSize, MPI_DOUBLE, strideIndex, MPI_COMM_WORLD);
				
				// perform the update of a selected sequence of columns of U
				updateUColsBlock(k, range);				
			}
		} // end of LU factorization part

		if (lastRow < max1) { // conditional execution of the SAXPY task
			
			// determine which process will provide the shared lBlock section
			int blockNo = range.min / blockStride;
			int blockIndex = range.min % blockStride;
			int strideIndex = blockIndex / blockSize;

			// perform the l-Block part generation step by copying in recently updated part of L
			if (strideIndex == rank) {
				copyUpdatedLBlock(range);
			}

			// do an MPI broadcast to share the l-Block among all processes
			int lBlockLength = lBlockDims[0].length * lBlockDims[1].length; 
			MPI_Bcast(lBlock, lBlockLength, MPI_DOUBLE, strideIndex, MPI_COMM_WORLD);

			// perform the saxpy computation
			saxpy(range);
		
		} // end of SAXPY part
	}

	//-------------------------------------------------------------- Computation Ends

	// write outputs to files
	if (fileIOMode) {
		writeUandLToFile();
		if (rank == 0) writePToFile();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	// end timer
	struct timeval end;
        gettimeofday(&end, NULL);
	if (rank == 0 && !fileIOMode) {
		double executionTime = ((end.tv_sec + end.tv_usec / 1000000.0)
				- (start.tv_sec + start.tv_usec / 1000000.0));
		cout << "Execution time: " << executionTime << " Seconds\n";
	}

	// do MPI teardown
	MPI_Finalize();
	return 0;
}



