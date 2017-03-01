#include <mpi.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

namespace mpi_monte {

//------------------------------------------------------------------- Supporting structures

typedef struct {
        int top;
        int left;
        int right;
        int bottom;
} Rectangle;


} // namespace ends

using namespace mpi_monte;

int mainMMonte(int argc, char *argv[]) {

        // do MPI intialization
	int rank, processCount;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processCount);

	if (argc < 4) {
                std::cout << "provide cell length, grid dimension in number of cells, and points";
                std::cout << " to be generated per cell\n";
		MPI_Finalize();
                std::exit(EXIT_FAILURE);
        }

	// start timer
        struct timeval start;
        gettimeofday(&start, NULL);

	// parse command line arguments
	int cell_length = atoi(argv[1]);
        int grid_dimension = atoi(argv[2]);
        int points_per_cell = atoi(argv[3]);

	// initialize the random number generator
	srand(time(NULL));

	// let different MPI processes estimate different stripes of rows of the grid block and accumulate their estimates
        double local_estimate = 0.0;
	double cell_size = cell_length * cell_length;
        for (int y = rank; y < grid_dimension; y += processCount) {
                for (int x = 0; x < grid_dimension; x++) {

                        // determine the grid cell boundaries
                        Rectangle cell;
                        cell.top = cell_length * (y + 1) - 1;
                        cell.bottom = cell_length * y;
                        cell.left = cell_length * x;
                        cell.right = cell_length * (x + 1) - 1;

                        // do appropriate number of random sampling
                        int points_inside = 0;
                        for (int s = 0; s < points_per_cell; s++) {
                                int x = rand() % cell_length + cell.left;
                                int y = rand() % cell_length + cell.bottom;
                                double result = 10 * sin(pow(x, 2)) + 50 * cos(pow(y, 3));
                                if (result <= 0.0) {
                                        points_inside++;
                                }
                        }

                        // calculate the subarea estimate for the grid cell and accumulate estimates
                       	double cell_area_estimate = cell_size * points_inside / points_per_cell;
			local_estimate += cell_area_estimate;
                }
        }

	// let the processes reduce their local estimates into a final area estimates that the first process will print
	double area_estimate = 0.0;
	int status = MPI_Reduce(&local_estimate, &area_estimate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (status != MPI_SUCCESS) {
		std::cout << rank << ": could not participate in the reduction\n";
		std::exit(EXIT_FAILURE);
	}

	 // end timer
        struct timeval end;
        gettimeofday(&end, NULL);
        if (rank == 0) {
                double executionTime = ((end.tv_sec + end.tv_usec / 1000000.0)
                                - (start.tv_sec + start.tv_usec / 1000000.0));
                std::cout << "Execution time: " << executionTime << " Seconds\n";
		std::cout << "Estimated area under the curve: " << area_estimate << "\n";
        }

	// do MPI teardown
        MPI_Finalize();
        return 0;
}
