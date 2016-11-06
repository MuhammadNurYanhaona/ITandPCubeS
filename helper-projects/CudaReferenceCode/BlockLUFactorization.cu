#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include "cublas_v2.h"

using namespace std;

// CUDA constants
#define WARP_SIZE 32
#define WARP_COUNT 16
#define BLOCK_COUNT 13

// Size of each matrix block for dual space computation
#define BLOCK_A_HEIGHT 32
#define CHUNK_SIZE 32
#define BLOCK_B_WIDTH 32

typedef struct {
	int dimension;
} ArrayMetadata1D;

typedef struct {
        int dimension1;
        int dimension2;
        __device__ int getSize() {
                return dimension1 * dimension2;
        }
} ArrayMetadata2D;

typedef struct {
	int blockSize;
	int blocksInStride;
} BlockStrideConfig;

typedef struct {
	int rowStart;
	int rowEnd;
	int colStart;
	int colEnd;
} IndexRange;


// -------------------------------------------------------------------------  Host function definitions
void die(const char *error);
void check_error(cudaError e);
void check_error(cublasStatus_t stat);
void loadOrInitializeA(ArrayMetadata2D A_MD, double *A);
void selectPivot(cublasHandle_t handle, int k, ArrayMetadata2D U_MD, double *U, int *P);

// -------------------------------------------------------------------------- CUDA function definitions
__global__ void prepare(BlockStrideConfig strideConfig, ArrayMetadata2D A_MD, double *A, 
			double *U, ArrayMetadata2D L_MD, double *L);
__global__ void interchangeRows(BlockStrideConfig strideConfig, int k, int pivot, 
			ArrayMetadata2D U_MD, double *U, ArrayMetadata2D L_MD, double *L);
__global__ void updateLower(int k, ArrayMetadata2D U_MD, double *U, double *L);
__global__ void updateUpperParts(int k, int min, int max, 
			double *lColumn, double *lRow, 
			ArrayMetadata2D U_MD, double *U);
__global__ void saxpy(const double *A_GPU, const double *B_GPU, double *C_GPU,
		ArrayMetadata2D A_MD, ArrayMetadata2D B_MD, ArrayMetadata2D C_MD,
		IndexRange A_Range, IndexRange B_Range, IndexRange C_Range);

// ----------------------------------------------------------------------------------------------- Main
int main(int argc, char *argv[]) {

	if (argc < 4) {
		cout << "Not enough arguments. Matrix length (1); "; 
		cout << "stride block size (2) and block count (3) are needed\n";
		exit(-1);
	}
	
	double *A, *U, *L; 
	int *P;
	ArrayMetadata2D A_MD, U_MD, L_MD;
	ArrayMetadata1D P_MD;
	
	A_MD.dimension1 = atoi(argv[1]);
	A_MD.dimension2 = A_MD.dimension1;

	BlockStrideConfig strideConfig;
	strideConfig.blockSize = atoi(argv[2]);
	strideConfig.blocksInStride = atoi(argv[3]);

	//----------------------------------------------------------------------------- Initialization
	clock_t start = clock();
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(double);
        check_error(cudaMalloc((void **) &A, sizeofA));
	U_MD.dimension1 = A_MD.dimension2;
	U_MD.dimension2 = A_MD.dimension1;
	double *U_CPU = (double *) malloc(sizeofA);
        check_error(cudaMalloc((void **) &U, sizeofA));
	L_MD = U_MD;
	double *L_CPU = (double *) malloc(sizeofA);
        check_error(cudaMalloc((void **) &L, sizeofA));
	P_MD.dimension = A_MD.dimension1;
	P = (int *) malloc(P_MD.dimension * sizeof(int));
	loadOrInitializeA(A_MD, A);

	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
    	cublasHandle_t handle;
	check_error(cublasCreate(&handle));
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

	clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Allocation time: " << elapsed << " seconds \n";
	
	start = clock();
	//-------------------------------------------------------------------------------- Computation
        
	prepare<<< BLOCK_COUNT, threadsPerBlock >>>(strideConfig, A_MD, A, U, L_MD, L);
	
	for (int r = 0; r < A_MD.dimension1; r += strideConfig.blockSize) {

		int min = r;
		int max = r + strideConfig.blockSize - 1;
		if (max >= A_MD.dimension1) max = A_MD.dimension1 - 1;

		for (int k = min; k <= max; k++) {
			selectPivot(handle, k, U_MD, U, P);
			cudaThreadSynchronize();
			if (P[k] != k) {
				interchangeRows<<< BLOCK_COUNT, threadsPerBlock >>>
						(strideConfig, k, P[k], U_MD, U, L_MD, L);
			}
			updateLower<<< BLOCK_COUNT, threadsPerBlock >>>(k, U_MD, U, L);

			double *lColumn = L + k; 
			double *lRow = L + k * L_MD.dimension2;
			
			updateUpperParts <<< BLOCK_COUNT, threadsPerBlock >>>
					(k, min, max, lColumn, lRow, U_MD, U);

		}
		if (max < A_MD.dimension1) {
			
			IndexRange aRange, bRange, cRange;
			
			aRange.rowStart = max + 1;
			aRange.rowEnd = U_MD.dimension1;
			aRange.colStart = min;
			aRange.colEnd = max;

			bRange.rowStart = min;
			bRange.rowEnd = max;
			bRange.colStart = max + 1;
			bRange.colEnd = L_MD.dimension2;

			cRange.rowStart = max + 1;
			cRange.rowEnd = U_MD.dimension1;
			cRange.colStart = max + 1;
			cRange.colEnd = U_MD.dimension2;

			saxpy <<< BLOCK_COUNT, threadsPerBlock >>>
					(U, L, U, U_MD, L_MD, U_MD, aRange, bRange, cRange);	
		}
	}

       	check_error(cudaGetLastError());
	cudaThreadSynchronize();
	
        check_error(cudaMemcpy(U_CPU, U, sizeofA, cudaMemcpyDeviceToHost));
        check_error(cudaMemcpy(L_CPU, L, sizeofA, cudaMemcpyDeviceToHost));

	//------------------------------------------------------------------------------------ Cleanup
	end = clock();
        elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Execution time: " << elapsed << " seconds \n";
	cublasDestroy(handle);

	return 0;
}

void loadOrInitializeA(ArrayMetadata2D A_MD, double *A) {

        size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(double);
        double *A_CPU = (double*) malloc(sizeofA);

        srand(time(NULL));
        for (int i = 0; i < A_MD.dimension1; i++)
                for (int j = 0; j < A_MD.dimension2; j++) {
                        int index = i * A_MD.dimension2 + j;
                        int value = rand();
			if (value == 0) value = i + j + 1;
			A_CPU[index] = value;
                }

        check_error(cudaMemcpyAsync(A, A_CPU, sizeofA, cudaMemcpyHostToDevice, 0));
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA Error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

// If any CUBLAS helper function fails then exit program
void check_error(cublasStatus_t stat) {
        if (stat != CUBLAS_STATUS_SUCCESS) {
                printf("\nCUBLAS Error\n");
                exit(1);
        }
}

__global__ void prepare(BlockStrideConfig strideConfig, ArrayMetadata2D A_MD, double *A, 
			double *U, ArrayMetadata2D L_MD, double *L) {
	
	int blockSize = strideConfig.blockSize;
	int blockCount = strideConfig.blocksInStride;

	// determine the IDs of different levels in the GPU
        int threadId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
        int smId = blockIdx.x;

	// determine the number of steps need to strided
	int singleStrideCover = blockSize * blockCount;
	int strideCount = (A_MD.dimension2 + singleStrideCover - 1) / singleStrideCover;

	// determine the nubmer of blocks to be handled by current SM
	if (smId >= blockCount) return;
	int blocksPerSM = (blockCount + BLOCK_COUNT - 1) / BLOCK_COUNT;
	int smBlocksStart = blocksPerSM * smId;
	int smBlocksEnd = blocksPerSM + smBlocksStart - 1;
	if (smBlocksEnd >= blockCount) smBlocksEnd = blockCount - 1;
	int smBlocks = smBlocksEnd - smBlocksStart + 1;

	// allocate a pannel for temporary storing A parts in the memory during initialization of U
	__shared__ double A_parts[WARP_SIZE][WARP_SIZE];

	//-------------------------------------------------------------------------------- Prepare U
	// process only WARP_SIZE rows of A at a time due to shared memory limitation
	for (int row = 0; row < A_MD.dimension1; row += WARP_SIZE) {
		int rowsStart = row;
		int rowsEnd = row + WARP_SIZE - 1;
		if (rowsEnd >= A_MD.dimension1) rowsEnd = A_MD.dimension1 - 1;
		// all warps stride from begin to end over the columns of A
		for (int stride = 0; stride < strideCount; stride++) {
			int startIndex = stride * singleStrideCover + smBlocksStart * blockSize;
			int endIndex = startIndex + blockSize * smBlocks - 1;
			if (endIndex >= A_MD.dimension2) endIndex = A_MD.dimension2 - 1;
			// process WARP_SIZE columns of a stride of A at a time
			for (int column = startIndex; column <= endIndex; column += WARP_SIZE) {
				// distribute the WARP_SIZE rows to different warps for read
				for (int r = rowsStart + warpId; r <= rowsEnd; r += WARP_COUNT) {
					// make different threads of a warp read different columns of A
					if (column + threadId <= endIndex) {
						int index = column + threadId + r * A_MD.dimension2;
						A_parts[threadId][r - rowsStart] = A[index];
					}
				}
				__syncthreads();
				// distribute now locally stored rows of U to different warps for write
				int uRowsStart = column;
				int uRowsEnd = column + WARP_SIZE - 1;
				if (uRowsEnd > endIndex) uRowsEnd = endIndex;
				for (int r = uRowsStart + warpId; r <= uRowsEnd; r += WARP_COUNT) {
					// make different threads of a warp write different columns of U
					if (rowsStart + threadId <= rowsEnd) {
						int index = rowsStart + threadId + r * A_MD.dimension1;
						U[index] = A_parts[r - uRowsStart][threadId];
					}
				}  
			}
		}
	}		
	
	//-------------------------------------------------------------------------------- Prepare L
	// all warps stride the rows of L sequentially
	for (int stride = 0; stride < strideCount; stride++) {
		int startIndex = stride * singleStrideCover + smBlocksStart * blockSize;
		int endIndex = startIndex + blockSize * smBlocks - 1;
		if (endIndex >= L_MD.dimension1) endIndex = L_MD.dimension1 - 1;
		// distribute the rows of L to different warps
		for (int row = startIndex + warpId; row <= endIndex; row += WARP_COUNT) {
			// distribute the columns of L to different threads for write 
			for (int column = threadId; column < L_MD.dimension2; column += WARP_SIZE) {
				int index = row * L_MD.dimension2 + column;
				L[index] = (column == row) ? 1 : 0;
			}
		}
	}
}

// --------------------------------------------------------------------------- Select and store pivot
void selectPivot(cublasHandle_t handle, int k, ArrayMetadata2D U_MD, double *U, int *P) {
	const double *U_row = U + k * U_MD.dimension2 + k;
	int columnCount = U_MD.dimension2 - k;
	int *P_column = P + k; 
	check_error(cublasIdamax(handle, columnCount, U_row, 1, P_column));
}

// ---------------------------------------------------------------- Intechange current and pivot rows
__global__ void interchangeRows(BlockStrideConfig strideConfig, int k, int pivot, 
			ArrayMetadata2D U_MD, double *U, ArrayMetadata2D L_MD, double *L) {	

	int blockSize = strideConfig.blockSize;
	int blockCount = strideConfig.blocksInStride;

	// determine the IDs of different levels in the GPU
        int threadId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
        int smId = blockIdx.x;

	// determine the number of steps need to strided
	int singleStrideCover = blockSize * blockCount;
	int strideCount = (U_MD.dimension1 + singleStrideCover - 1) / singleStrideCover;

	// determine the nubmer of blocks to be handled by current SM
	if (smId >= blockCount) return;
	int blocksPerSM = (blockCount + BLOCK_COUNT - 1) / BLOCK_COUNT;
	int smBlocksStart = blocksPerSM * smId;
	int smBlocksEnd = blocksPerSM + smBlocksStart - 1;
	if (smBlocksEnd >= blockCount) smBlocksEnd = blockCount - 1;
	int smBlocks = smBlocksEnd - smBlocksStart + 1;


// ---------------------------------------------------------------------------------------- Update U
	int cutOffStride = k / singleStrideCover;
	// distribute the strides among warps
	for (int stride = cutOffStride + warpId; stride < strideCount; stride += WARP_COUNT) {
		int startIndex = stride * singleStrideCover + smBlocksStart * blockSize;
		int endIndex = startIndex + blockSize * smBlocks - 1;
		if (startIndex < k) startIndex = k;
		if (endIndex >= U_MD.dimension1) endIndex = U_MD.dimension1 - 1;
		// let different threads interchange elements from different rows
		for (int row = startIndex + threadId; row <= endIndex; row += WARP_SIZE) {
			int currentKthIndex = row * U_MD.dimension2 + k;
			double currentKthValue = U[currentKthIndex];
			int currentPivotIndex = row * U_MD.dimension2 + pivot;
			U[currentKthIndex] = U[currentPivotIndex];
			U[currentPivotIndex] = currentKthValue;
		}
	}
	
// ---------------------------------------------------------------------------------------- Update L
	// distribute the strides among warps
	for (int stride = warpId; stride <= cutOffStride; stride += WARP_COUNT) {
		int startIndex = stride * singleStrideCover + smBlocksStart * blockSize;
		if (startIndex >= k) break;
		int endIndex = startIndex + blockSize * smBlocks - 1;
		if (endIndex >= k) endIndex = k - 1;
		// let different threads interchange elements from different rows
		for (int row = startIndex + threadId; row <= endIndex; row += WARP_SIZE) {
			int currentKthIndex = row * L_MD.dimension2 + k;
			double currentKthValue = L[currentKthIndex];
			int currentPivotIndex = row * L_MD.dimension2 + pivot;
			L[currentKthIndex] = L[currentPivotIndex];
			L[currentPivotIndex] = currentKthValue;
		}
	}
}

// -------------------------------------------------------------------------- Create L's Kth column
__global__ void updateLower(int k, ArrayMetadata2D U_MD, double *U, double *L) {
	
	int pivotIndex = k * U_MD.dimension2 + k;
	int startingIndex = pivotIndex + 1;
	int elementCount = U_MD.dimension2 - k - 1;
	int endIndex = pivotIndex + elementCount;

	int threadId = threadIdx.x;
	int smId = blockIdx.x; 	
        int warpId = threadIdx.x / WARP_SIZE;
	int stripeWarpId = warpId * WARP_COUNT + smId;
	int stripeThreadId = stripeWarpId * WARP_SIZE + threadId;
	int stripeLength = WARP_COUNT * WARP_SIZE * BLOCK_COUNT;

	double pivotElement = U[pivotIndex];
	for (int element = startingIndex + stripeThreadId; element <= endIndex; element += stripeLength) {
		L[element] = U[element] / pivotElement;
	}
}

// --------------------------------------------------------- Update blocks of rows and columns of U
__global__ void updateUpperParts(int k, int min, int max, 
			double *lColumn, double *lRow, 
			ArrayMetadata2D U_MD, double *U) {
	
        // determine the IDs and count of different levels in the GPU
        int smId = blockIdx.x;
	int smThreadId = threadIdx.x;
	int smThreadCount = WARP_COUNT * WARP_SIZE;
	int totalThreads = smThreadCount * BLOCK_COUNT;

	// ------------------------------------------------first update the block of rows
	//	do {    u[i][j] = u[i][j] - l_row[j] * u[i][k] 
        //      } for i, j in u and i > k and i <= row_range.max and j > k

	// distribute different rows of U to different SMs
	for (int row = k + 1 + smId; row <= max; row += BLOCK_COUNT) {

		int row_index = row * U_MD.dimension2;
		int pivot_index = row * U_MD.dimension2 + k;

		// distribute different columns to different threads of the SM
		for (int column = k + 1 + smThreadId; column < U_MD.dimension2 - 1; column += smThreadCount) {
			U[row_index + column] -= lRow[column] * U[pivot_index];
		}
	}

	// --------------------------------------------second update the block of columns
	// 	do {    u[i][k] = u[i][k] - u[i][j] * l_column[j]
	// 	} for i, j in u and i > row_range.max and j >= row_range.min and j < k

	// distribute different block of rows of U to different SMs
	int startRowBlock = max + 1 + smId * smThreadCount;
	for (int rowBlock = startRowBlock; rowBlock < U_MD.dimension1; rowBlock += totalThreads) {
		
		// let different threads handle different rows of a block within an SM
		int row = rowBlock + smThreadId;
		if (row >= U_MD.dimension1) continue;

		int row_index = row * U_MD.dimension2;
		int pivot_index = row * U_MD.dimension2 + k;
		double lColumnElement = *(lColumn + (row * U_MD.dimension2));
		for (int column = min; column < k; column++) {
			U[pivot_index] -= U[row_index + column] * lColumnElement;
		}
	}	
}

// --------------------------------------------------------------------- Update remaining rows of U
__global__ void saxpy(const double *A_GPU, const double *B_GPU, double *C_GPU,
		ArrayMetadata2D A_MD, ArrayMetadata2D B_MD, ArrayMetadata2D C_MD,
		IndexRange A_Range, IndexRange B_Range, IndexRange C_Range) {

	int threadId = threadIdx.x % WARP_SIZE;
	int warpId = threadIdx.x / WARP_SIZE;
	int blockId = blockIdx.x;

	// two shared memory 2D arrays for storing chunks from a number of rows and columns of A and B respectively
	__shared__ double A_parts[BLOCK_A_HEIGHT][CHUNK_SIZE];
	__shared__ double B_parts[CHUNK_SIZE][BLOCK_B_WIDTH];

	// one shared memory array where each warp stores partial D[i][j][l] += A[i][k] * B[j][k] values that at the 
	// end get accumulated into C[i][j]
	__shared__ double D_partialResults[BLOCK_A_HEIGHT][BLOCK_B_WIDTH];

	// allocate a consecutive number of rows to each thread block; each will process all columns
	int rowsPerBlock = (A_Range.rowEnd - A_Range.rowStart + 1) / BLOCK_COUNT;
	int extraRows = (A_Range.rowEnd - A_Range.rowStart + 1) % BLOCK_COUNT;
	int rowsBegin = A_Range.rowStart + rowsPerBlock * blockId;
	int rowsEnd = rowsBegin + rowsPerBlock - 1;
	rowsEnd += (blockId < BLOCK_COUNT - 1) ? 0 : extraRows; // last block is taking care of some extra rows;
								// ideally, 10these should also be distributed 

	int commonDimensionLength = A_Range.colEnd - A_Range.colStart + 1;

	// at a time BLOCK_A_HEIGHT number of rows are processed
	for (int i = rowsBegin; i <= rowsEnd; i += BLOCK_A_HEIGHT) { 

		// at a time BLOCK_B_WIDTH number of columns are processed
		for (int j = B_Range.colStart; j <= B_Range.colEnd; j+= BLOCK_B_WIDTH) { 
				
			// check how many rows and columns of local A and B parts are valid
			int lastValidRow = i + BLOCK_A_HEIGHT - 1;
			if (lastValidRow > rowsEnd) lastValidRow = rowsEnd;
			int rowCount = lastValidRow - i + 1;
			int lastValidColumn = j + BLOCK_B_WIDTH - 1;
			if (lastValidColumn > B_Range.colEnd) lastValidColumn = B_Range.colEnd - 1;
			int columnCount = lastValidColumn - j + 1; 	
		
			// reset D[i][j]'s to zeros to accumate results for new a (i, j) indices
			for (int row = warpId; row < BLOCK_A_HEIGHT; row += WARP_COUNT) {
				for (int column = threadId; column < BLOCK_B_WIDTH; column += WARP_SIZE) {
					D_partialResults[row][column] = 0;
				}
			}
		
			// For each row and column, only a section of chunk-size would be downloaded once as instructed
			// by the programmer through sub-partition specification
			for (int k = 0; k < commonDimensionLength; k += CHUNK_SIZE) {
				__syncthreads(); 

				// Cleanup operation .................................................................
				// cleanup old/invalid A_parts and B_parts from shared memory
				for (int row = warpId; row < BLOCK_A_HEIGHT; row += WARP_COUNT) {
					for (int cleanupIndex = threadId; cleanupIndex < CHUNK_SIZE; 
							cleanupIndex += WARP_SIZE) {
						A_parts[row][cleanupIndex] = 0;
					}
				}
				for (int row = warpId; row < CHUNK_SIZE; row += WARP_COUNT) {
					for (int cleanupIndex = threadId; cleanupIndex < BLOCK_B_WIDTH; 
							cleanupIndex += WARP_SIZE) {
						B_parts[row][cleanupIndex] = 0;
					}
				}
				
				// Data read from global to shared memory .............................................

				// determine A's row and column boundaries
				int beginARow = i;
				int endARow = (i + BLOCK_A_HEIGHT - 1) > rowsEnd ?
						rowsEnd : beginARow + BLOCK_A_HEIGHT - 1;
				int beginAColumn = k;
				int endAColumn = (k + CHUNK_SIZE - 1) >= commonDimensionLength ?
						commonDimensionLength - 1 : beginAColumn + CHUNK_SIZE - 1;
				beginAColumn += A_Range.colStart;
				endAColumn += A_Range.colStart;		
		
				// download a section of A; differnt warps download different rows	
				for(int rowForCurrentWarp = beginARow + warpId; rowForCurrentWarp <= endARow; 
					rowForCurrentWarp += WARP_COUNT) {
					int localRowIndex = rowForCurrentWarp - beginARow;
					int globalRowStart = rowForCurrentWarp * A_MD.dimension2;
					// different threads download different elements of A from the global memory
					for (int elementIndex = beginAColumn + threadId; 
							elementIndex <= endAColumn; elementIndex += WARP_SIZE) {
						A_parts[localRowIndex][elementIndex - beginAColumn] 
							= A_GPU[globalRowStart + elementIndex];
					}	
				}

				// determine B's row and column boundaries
				int beginBRow = k;
				int endBRow = (k + CHUNK_SIZE - 1) >= commonDimensionLength ?
						commonDimensionLength - 1 : beginBRow + CHUNK_SIZE - 1;
				beginBRow += B_Range.rowStart;
				endBRow += B_Range.rowStart;
				int beginBColumn = j;
				int endBColumn = (j + BLOCK_B_WIDTH - 1) >= B_MD.dimension2 ?
						B_MD.dimension2 - 1 : beginBColumn + BLOCK_B_WIDTH - 1;
	
				// download a section of B; different warps download different rows
				for (int rowForCurrentWarp = beginBRow + warpId; rowForCurrentWarp <= endBRow; 
						rowForCurrentWarp += WARP_COUNT) {
					int localRowIndex = rowForCurrentWarp - beginBRow;
					int globalRowStart = rowForCurrentWarp * B_MD.dimension2;
					// different threads download different elements of B from the global memory
					for (int elementIndex = beginBColumn + threadId; 
							elementIndex <= endBColumn; elementIndex += WARP_SIZE) {
						B_parts[localRowIndex][elementIndex - beginBColumn] 
							= B_GPU[globalRowStart + elementIndex];
					}	
				}
				//__threadfence_block();
				__syncthreads(); // do a sync to make A and B parts available to all threads in the SM
				
				// Block matrix multiplication kernel .............................................
				// different threads take care of different B columns
				for (int c = threadId; c < columnCount; c+= WARP_SIZE) {
					// different warps take care of different rows of A
					for (int r = warpId; r < rowCount; r += WARP_COUNT) {
						// all threads go through each values of the common dimension
						for (int e = 0; e < CHUNK_SIZE; e++) {
							D_partialResults[r][c] += A_parts[r][e] * B_parts[e][c];
						}
					}	
				}	 
			} // loop over k's ends

			//__threadfence_block();
			__syncthreads(); // do a sync to make updated D == C available to all threads
		
			// Data upload from shared to global memory .................................................
			// let different warps upload different rows and different threads within each warp
			// upload different columns of C
			for (int row = warpId; row < rowCount; row += WARP_COUNT) {
				int rowIndex = i + row;
				for (int column = threadId; column < columnCount; column += WARP_SIZE) {
					int columnIndex = j + column;
					int C_index = rowIndex * C_MD.dimension2 + columnIndex;
					C_GPU[C_index] -= D_partialResults[row][column];
				}
			}
		} // loop over j's ends
	} // loop over i's ends
}
