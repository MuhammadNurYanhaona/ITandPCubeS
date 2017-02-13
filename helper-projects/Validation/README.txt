This project has several kinds of reference programs. These are as follows:

1. Sequential reference programs for timing a sequential code implementing an
   algorithm equivalent to an IT code.
2. Validator reference programs for validating the correctness of a IT code by
   checking the IT output files against the output of a sequential reference.
3. MPI reference programs for performance timing of optimized reference MPI 
   implementations. (These are pure MPI codes.)
4. Pthread reference programs for doing the same kind of performance analysis.
5. OpenMP reference programs that implement algorithmic alternative to a IT code
   that is simpler and natural for an OpenMP programmer to implement.

To generate an executable that will run a specific reference program do as 
follows:

1. First go to the specific folder for the reference program type.
2. Open the reference program and you will see the the main function is renamed
   to have something appended to main. Rename the function to main.
3. Make the project with specific Makefile for the reference program type using
   the command
	make -f $Makefile-Name
4. The executable name has the following format
	$typeName-ref

An example:
------------
Suppose we want to run the Block LUF MPI program for performance timing then
1. go to mpi folder
2. open the mpiBLUF.cpp file and rename the main function's name properly
3. come out of the mpi folder
4. run make -f MPI-Makefile
5. run the 'mpi-ref' executable using some MPI utility 


** each make-file has a 'clean' target that you can use to remove the object files
after you are done with your experiment.

** If you want some parallel reference program to read input and write output then
use the fileUtility.h and stream.h services in the program. If the program already
has support for optional file input-output then you might have to change the 
static paths mentioned in the program for input/output files to some paths valid
to your experimental settings.
 
