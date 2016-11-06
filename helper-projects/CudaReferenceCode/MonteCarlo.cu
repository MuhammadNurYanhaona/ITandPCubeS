#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

using namespace std;

// CUDA settings
#define WARP_SIZE 32
#define WARP_COUNT 16
#define BLOCK_COUNT 13

class Rectangle {
  public:
        int top;
        int bottom;
        int left;
        int right;
         __host__ __device__ Rectangle() {
                top = 0;
                bottom = 0;
                left = 0;
                right = 0;
        }
};

void die(const char *error);
void check_error(cudaError e);
__global__ void monteCarloKernel(double *subarea_estimates, 
		int cell_length,
                int grid_dimension, 
		int points_per_cell);

double *subarea_estimates_CPU, *subarea_estimates_GPU;

int main(int argc, char **argv) {

        if (argc < 4) {
                cout << "You need to provide three arguments\n";
                cout << "The length of each square grid cell\n";
                cout << "The dimension of the grid in terms of cell count along a dimension\n";
                cout << "The number of random samples to be generated per cell\n";
        }

        int cell_length = atoi(argv[1]);
        int grid_dimension = atoi(argv[2]);
        int points_per_cell = atoi(argv[3]);
	cout << "Cell length: " << cell_length << "\n";
	cout << "Grid dimension: " << grid_dimension << " by " << grid_dimension << "\n";
	cout << "Random samples per cell: " << points_per_cell << "\n";

        clock_t start = clock();

        long int grid_size = grid_dimension * grid_dimension * sizeof(double);
        subarea_estimates_CPU = (double *) malloc(grid_size);
        check_error(cudaMalloc((void **) &subarea_estimates_GPU, grid_size));

        clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Allocation time: " << elapsed << " seconds \n";

        start = clock();
        int threadsPerBlock = WARP_SIZE * WARP_COUNT;
        monteCarloKernel<<< BLOCK_COUNT, threadsPerBlock >>>
                        (subarea_estimates_GPU, cell_length, grid_dimension, points_per_cell);

        cudaThreadSynchronize();
        check_error(cudaGetLastError());
        check_error(cudaMemcpy(subarea_estimates_CPU, subarea_estimates_GPU, grid_size, cudaMemcpyDeviceToHost));

	double area_estimate = 0.0;
	for (int i = 0; i < grid_dimension; i++) {
		for (int j = 0; j < grid_dimension; j++) {
			area_estimate += subarea_estimates_CPU[i * grid_dimension + j];
		}	
	}
	cout << "Area under the curve is: " << area_estimate << "\n";

        end = clock();
        elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Execution time: " << elapsed << " seconds \n";
        return 0;
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

__global__ void monteCarloKernel(double *subarea_estimates, 
		int cell_length,
                int grid_dimension, 
		int points_per_cell) {

        int threadId = threadIdx.x;
	int threads_per_block = WARP_SIZE * WARP_COUNT;
        int blockId = blockIdx.x;
	double cell_size = cell_length * cell_length;

	// distribute the horizontal axis points among the SMs
	for (int x = blockId; x < grid_dimension; x += BLOCK_COUNT) {
		// distribute the vertical axis points among the threads of an SM
		for (int y = threadId; y < grid_dimension; y += threads_per_block) {

			// determine grid cell boundaries
			Rectangle cell;
			cell.left = cell_length * x;
			cell.right = cell_length * (x + 1) - 1;
			cell.bottom = cell_length * y;
			cell.top = cell_length * ( y + 1) - 1;

			// initialze a random number generator
			curandState_t state;
			int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
			curand_init(clock64(), threadIndex, 0, &state);

			// perform the sampling trials
			int internal_points = 0;	
			for (int trial = 0; trial < points_per_cell; trial++) {
				
				// generate a point within the cell boundary and calculate
				// its position relative to the shape
				int xp = ((int) curand_uniform(&state)) 
						% cell_length + cell.left;
				int yp = ((int) curand_uniform(&state)) 
						% cell_length + cell.bottom;
				
				// tested polynomial is 10 sin x^2 + 50 cos y^3
				double x_2 = xp * xp;
				double y_3 = yp * yp * yp;
				double result = 10 * sin(x_2) + 50 * cos(y_3);
				if (result <= 0.0) {
					internal_points++; 
				}
			}

			// estimate the part of the polynomial within the grid cell
			long int index = y * grid_dimension + x;
			subarea_estimates[index] = cell_size * internal_points / points_per_cell;
		}
	}
}
