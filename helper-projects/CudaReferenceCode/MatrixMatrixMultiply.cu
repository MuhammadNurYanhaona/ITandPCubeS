#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>

using namespace std;

// CUDA settings
#define WARP_SIZE 32
#define WARP_COUNT 16
#define BLOCK_COUNT 13

// Size of each matrix block for dual space computation
#define BLOCK_A_HEIGHT 32
#define CHUNK_SIZE 32
#define BLOCK_B_WIDTH 32

typedef struct {
	int dimension1;
	int dimension2;	
        __device__ int getSize() {
                return dimension1 * dimension2;
        }
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory  
double *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
double *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------
void createMetadataForC();
void allocateAndInitializeAB();
void allocateC();
void die(const char *error); 
void check_error(cudaError e);
void computeCpuMMM();
void compareHostAndGpuOutput();
 

//----------------------------------- CUDA function definitions -----------------------------------

__global__ void matrixMultiplicationKernelDualSpace(const double *A_GPU, const double *B_GPU, double *C_GPU,
		ArrayMetadata2D A_MD, ArrayMetadata2D B_MD, ArrayMetadata2D C_MD);

int main(int argc, char **argv) {
	
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 1000;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;

	clock_t start = clock();	
	createMetadataForC();
	allocateAndInitializeAB();
	allocateC();
	clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Allocation time: " << elapsed << " seconds \n";

	start = clock();
	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
	matrixMultiplicationKernelDualSpace<<< BLOCK_COUNT, threadsPerBlock >>> 
			(A_GPU, B_GPU, C_GPU, A_MD, B_MD, C_MD);
	
	cudaThreadSynchronize();
	check_error(cudaGetLastError());
	
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(double);
	C_CPU = (double *) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
	
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	cout << "Execution time: " << elapsed << " seconds \n";
	return 0;
}

// gather and store the dimensionality information concerning both C and D
void createMetadataForC() {
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;
}

// initialize A and B using a random number generator then copy the two matrix in GPU memory
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(double);
	A = (double*) malloc(sizeofA);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++)
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 5); 
		}
	
	check_error(cudaMemcpyAsync(A_GPU, A, sizeofA, cudaMemcpyHostToDevice, 0));
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(double);
	B = (double*) malloc(sizeofB);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
  	for (int i = 0; i < B_MD.dimension1; i++)
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 5); 
		}
	
	check_error(cudaMemcpyAsync(B_GPU, B, sizeofB, cudaMemcpyHostToDevice, 0));
}

// allocate C in the device and hold a reference of it in the host memory for later use
void allocateC() {
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(double);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

__global__ void matrixMultiplicationKernelDualSpace(const double *A_GPU, const double *B_GPU, double *C_GPU,
		ArrayMetadata2D A_MD, ArrayMetadata2D B_MD, ArrayMetadata2D C_MD) {

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
	int rowsPerBlock = A_MD.dimension1 / BLOCK_COUNT;
	int extraRows = A_MD.dimension1 % BLOCK_COUNT;
	int rowsBegin = rowsPerBlock * blockId;
	int rowsEnd = rowsBegin + rowsPerBlock - 1;
	rowsEnd += (blockId < BLOCK_COUNT - 1) ? 0 : extraRows; // last block is taking care of some extra rows;
								// ideally, 10these should also be distributed 

	int commonDimensionLength = A_MD.dimension2;

	// at a time BLOCK_A_HEIGHT number of rows are processed
	for (int i = rowsBegin; i <= rowsEnd; i += BLOCK_A_HEIGHT) { 
		// at a time BLOCK_B_WIDTH number of columns are processed
		for (int j = 0; j < B_MD.dimension2; j+= BLOCK_B_WIDTH) { 
				
			// check how many rows and columns of local A and B parts are valid
			int lastValidRow = i + BLOCK_A_HEIGHT - 1;
			if (lastValidRow > rowsEnd) lastValidRow = rowsEnd;
			int rowCount = lastValidRow - i + 1;
			int lastValidColumn = j + BLOCK_B_WIDTH - 1;
			if (lastValidColumn >= B_MD.dimension2) lastValidColumn = B_MD.dimension2 - 1;
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
					C_GPU[C_index] = D_partialResults[row][column];
				}
			}
		} // loop over j's ends
	} // loop over i's ends
}

void computeCpuMMM() {

        // allocate the result matrix for the CPU computation
        size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(double);
        C = (double*) malloc(sizeofC);

        // compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
        for (int i = 0; i < A_MD.dimension1; i++) {
                int a_i = i * A_MD.dimension2;
                int c_i = i * C_MD.dimension2;
                for (int j = 0; j < B_MD.dimension2; j++) {
                        int c_index = c_i + j;
                        C[c_index] = 0;
                        for (int k = 0; k < B_MD.dimension1; k++) {
                                int a_index = a_i + k;
                                int b_index = k * B_MD.dimension2 + j;
                                C[c_index] += A[a_index] * B[b_index];
                        }
                }
        }
}

void compareHostAndGpuOutput() {
        int totalElements = C_MD.dimension1 * C_MD.dimension2;
        int missmatchCount = 0;
        for (int i = 0; i < totalElements; i++) {
                if (fabs(C[i] - C_CPU[i]) > 0.01) {
                        missmatchCount++;
                        printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
                }
        }
        if (missmatchCount > 0) {
                printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
        } else {
                printf("Computation is correct: CPU and GPU outputs match\n");
        }
}


