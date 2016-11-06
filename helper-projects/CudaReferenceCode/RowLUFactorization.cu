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
__global__ void updateUpper(BlockStrideConfig strideConfig, int k, double *lColumn, 
			ArrayMetadata2D U_MD, double *U);

// ----------------------------------------------------------------------------------------------- Main
int main(int argc, char *argv[]) {

	if (argc < 4) {
		cout << "Not enough arguments. Matrix length (1); "; 
		cout << "stride block size (2) and block count (3) are needed\n";
		exit(-1);
	}
	
	double *A, *U, *L;
	double *U_CPU, *L_CPU; 
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
        check_error(cudaMalloc((void **) &U, sizeofA));
	U_CPU = (double *) malloc(sizeofA);
	L_MD = U_MD;
        check_error(cudaMalloc((void **) &L, sizeofA));
	L_CPU = (double *) malloc(sizeofA);
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

	for (int k = 0; k < A_MD.dimension1; k++) {
		selectPivot(handle, k, U_MD, U, P);
		cudaThreadSynchronize();
		if (P[k] != k) {
			interchangeRows<<< BLOCK_COUNT, threadsPerBlock >>>(strideConfig, k, P[k], U_MD, U, L_MD, L);
		}
		updateLower<<< BLOCK_COUNT, threadsPerBlock >>>(k, U_MD, U, L);
		double *lColumn = L + k * L_MD.dimension2; 
		updateUpper<<< BLOCK_COUNT, threadsPerBlock>>>(strideConfig, k, lColumn, U_MD, U);
	}
       	check_error(cudaGetLastError());
	cudaThreadSynchronize();

	//----------------------------------------------------------------------------- Data stage out

	check_error(cudaMemcpy(L_CPU, L, sizeofA, cudaMemcpyDeviceToHost));
	check_error(cudaMemcpy(U_CPU, U, sizeofA, cudaMemcpyDeviceToHost));

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

// ------------------------------------------------------------------------------- Update rows of U
__global__ void updateUpper(BlockStrideConfig strideConfig, int k, double *lColumn, ArrayMetadata2D U_MD, double *U) {
	
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

	// allocate a pannel for holding a portion of the pivot column during updating other columns of U  
	__shared__ double pivot_column_part[WARP_COUNT];

	int cutoffRow = k;
	int cutoffColumn = k + 1;
	int cutoffStride = cutoffRow / singleStrideCover;

	// all warps strides from beginning to end of the rows of U from the cutoff point
	for (int stride = cutoffStride; stride < strideCount; stride++) {
		int startIndex = stride * singleStrideCover + smBlocksStart * blockSize;
		int endIndex = startIndex + blockSize * smBlocks - 1;
		if (startIndex < cutoffRow) startIndex = cutoffRow;
		if (endIndex >= U_MD.dimension1) endIndex = U_MD.dimension1 - 1;
		// process only WARP_COUNT rows of a stride at a time
		for (int row = startIndex; row <= endIndex; row += WARP_COUNT) {
			// let the first warp load the needed portion of U's pivot column into shared memory
			if (warpId == 0) {
				if (threadId < WARP_COUNT && startIndex + threadId <= endIndex) {
					int index = k + (startIndex + threadId) * U_MD.dimension2;
					pivot_column_part[threadId] = U[index];
				}
			}
			__syncthreads();
			// let different warps update different rows of U
			int warpRowId = row + warpId;
			if (warpRowId > endIndex) break;
			// distribute the columns among the threads of the warp
			for (int column = cutoffColumn + threadId; column < U_MD.dimension2; column += WARP_SIZE) {
				int index = warpRowId * U_MD.dimension2 + column;
				U[index] -= pivot_column_part[warpRowId - row] * lColumn[column];
			}
		}	
	} 
}




