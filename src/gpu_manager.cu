#include "gpu_manager.h"
#include "utils.h"
#include <stdlib.h>

GPUContext* initGPUContext(int gridHeight) {
    GPUContext* ctx = (GPUContext*)malloc(sizeof(GPUContext));
    
    CUDA_CHECK(cudaGetDeviceCount(&ctx->nDevices));
    
    if (ctx->nDevices < 1) {
        fprintf(stderr, "No CUDA devices found!\n");
        free(ctx);
        return NULL;
    }
    
    // Calculate dimensions
    ctx->localHeight = gridHeight / ctx->nDevices;
    ctx->haloSize = 1;
    ctx->localHeightWithHalo = ctx->localHeight + 2 * ctx->haloSize;
    
    // Allocate arrays for device resources
    ctx->d_grid = (unsigned char**)malloc(ctx->nDevices * sizeof(unsigned char*));
    ctx->d_grid_next = (unsigned char**)malloc(ctx->nDevices * sizeof(unsigned char*));
    ctx->streams = (cudaStream_t*)malloc(ctx->nDevices * sizeof(cudaStream_t));
    ctx->comms = (ncclComm_t*)malloc(ctx->nDevices * sizeof(ncclComm_t));
    
    // Initialize NCCL
    NCCL_CHECK(ncclCommInitAll(ctx->comms, ctx->nDevices, NULL));
    
    // Create streams
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&ctx->streams[i]));
    }
    
    return ctx;
}

void allocateGrids(GPUContext* ctx, int gridWidth) {
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&ctx->d_grid[i], 
                              gridWidth * ctx->localHeightWithHalo));
        CUDA_CHECK(cudaMalloc(&ctx->d_grid_next[i], 
                              gridWidth * ctx->localHeightWithHalo));
    }
}

void freeGrids(GPUContext* ctx) {
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(ctx->d_grid[i]));
        CUDA_CHECK(cudaFree(ctx->d_grid_next[i]));
    }
}

void destroyGPUContext(GPUContext* ctx) {
    if (!ctx) return;
    
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamDestroy(ctx->streams[i]));
        ncclCommDestroy(ctx->comms[i]);
    }
    
    free(ctx->d_grid);
    free(ctx->d_grid_next);
    free(ctx->streams);
    free(ctx->comms);
    free(ctx);
}
