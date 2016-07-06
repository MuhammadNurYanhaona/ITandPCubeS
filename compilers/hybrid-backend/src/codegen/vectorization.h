#ifndef _H_vectorization
#define _H_vectorization

// This header file hosts library routines that transform generated code for computation stages of a task
// into alternative code that embodies vectorization instruction. Thus the strategy adopted here is to 
// post-process the target code instead of dirctly spewing vectorized C/CUDA instructions from IT statements. 
// In the future, if we decide to do the latter then this library routines wont be needed anymore.

// This function changes sequential, possibly with multiple nestings, loop headers into alternative loop
// headers with embedded vectorization instructions. This transformation is only applicable if the target
// platform allows vectorization through loop index distribution (for example CUDA and OpenMP support this).
const char *vectorizeLoops(const char *originalLoopHeaders, int indentLevel);
 
#endif
