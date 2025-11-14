#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA kernel for Conway's Game of Life update
__global__ void gameOfLifeKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, 
    int localHeight, 
    int globalYOffset
);

#endif // KERNELS_CUH