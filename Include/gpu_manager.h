#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <cuda_runtime.h>
#include <nccl.h>

typedef struct {
    int nDevices;
    int localHeight;
    int haloSize;
    int localHeightWithHalo;
    
    unsigned char** d_grid;
    unsigned char** d_grid_next;
    cudaStream_t* streams;
    ncclComm_t* comms;
} GPUContext;

// Initialize GPU context
GPUContext* initGPUContext(int gridHeight);

// Cleanup GPU context
void destroyGPUContext(GPUContext* ctx);

// Allocate grid memory on all GPUs
void allocateGrids(GPUContext* ctx, int gridWidth);

// Free grid memory on all GPUs
void freeGrids(GPUContext* ctx);

#endif // GPU_MANAGER_H