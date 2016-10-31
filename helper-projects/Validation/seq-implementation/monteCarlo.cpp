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
#include <math.h>

using namespace std;

typedef struct {
	int top;
	int left;
	int right;
	int bottom;
} Rectangle;

int mainMonteCarlo(int argc, char *argv[]) {
	
	if (argc < 4) {
		std::cout << "provide cell length, grid dimension in number of cells, and points";
		std::cout << " to be generated per cell\n";
		std::exit(EXIT_FAILURE);
	}

	int cell_length = atoi(argv[1]);
	int grid_dimension = atoi(argv[2]);
	int points_per_cell = atoi(argv[3]);
	double cell_size = cell_length * cell_length;

	cout << "Cell length: " << cell_length << "\n";
	cout << "Grid dimension: " << grid_dimension << " by " << grid_dimension << "\n";
	cout << "Sample points per cell: " << points_per_cell << "\n";

	// starting execution timer clock
        struct timeval start;
        gettimeofday(&start, NULL);

	// initialize the random number generator
	srand(time(NULL));

	// initialize the area estimate variable;
	double estimate = 0.0;

	for (int x = 0; x < grid_dimension; x++) {
		for (int y = 0; y < grid_dimension; y++) {
			
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

			// calculate the subarea estimate for the grid cell
			double subarea = cell_size * points_inside / points_per_cell;

			// add the value to the total estimate
			estimate += subarea;
		}
	}

	//-------------------------------- calculate running time
        struct timeval end;
        gettimeofday(&end, NULL);
        double runningTime = ((end.tv_sec + end.tv_usec / 1000000.0)
                        - (start.tv_sec + start.tv_usec / 1000000.0));
        std::cout << "Sequential Execution Time: " << runningTime << " Seconds" << std::endl;
	
	return 0;
}
