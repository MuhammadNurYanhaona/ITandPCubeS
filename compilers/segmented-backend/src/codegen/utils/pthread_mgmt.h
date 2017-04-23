#ifndef _H_pthread_mgmt
#define _H_pthread_mgmt

/* This header file includes functions that aid in generating pthreads to run IT tasks. */

// PThreads library supports only one argument in the thread run function. Therefore we need to
// wrap up variables needed by the original run function into a new data structure that will be
// unpacked by the pthread run function. This function generates that structure.
void generateArgStructForPthreadRunFn(const char *taskName, const char *fileName);

// A PThreads run function separate from the run function generated for any IT task is needed as 
// the library requires the thread run function to confirm to a particular signature. The only
// thing this new function does in our case is that it unpacks elements from the argument and 
// calls the actual task run function with those parameters. 
void generatePThreadRunFn(const char *headerFile, const char *programFile, const char *initials);

#endif
