#ifndef _H_gpu_constant
#define _H_gpu_constant

// number of threads in a CUDA warp
#define WARP_SIZE 32

// Allocation of pointers from the dynamic shared memory pannel should always happen at some multiple of the
// following constant to avoid any alignment problem.  
#define MEMORY_PANNEL_ALIGNMENT_BOUNDARY 8

#endif
