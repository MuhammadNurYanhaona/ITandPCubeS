#include <omp.h>
#include <iostream>

int mainOmpTest() {

	int nthreads, tid;

	/* Fork a team of threads with each thread having a private tid variable */
	#pragma omp parallel private(tid)
	{
		/* Obtain and print thread id */
		tid = omp_get_thread_num();
		std::cout << "Hello World from thread = " <<  tid << '\n';

		/* Only master thread does this */
		if (tid == 0) {
			nthreads = omp_get_num_threads();
			std::cout << "Number of threads = " << nthreads << '\n';
		}
	}  /* All threads join master thread and terminate */

	return 0;
}
