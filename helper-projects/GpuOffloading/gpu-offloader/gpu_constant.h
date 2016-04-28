#ifndef _H_gpu_constant
#define _H_gpu_constant

/* The constants of this header file provide information about the GPU card. When including features of this project into 
 * the compiler, we need to generate these constants by parsing the PCubeS description of the GPU card, instead of using 
 * fixed values.
 */

// number of threads in a CUDA warp
#define WARP_SIZE 32

// number of warps to be employed per SM computation
#define WARP_COUNT 16

// number of SMs the GPU card has
#define BLOCK_COUNT 15

#endif
