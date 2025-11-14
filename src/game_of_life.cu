#include "game_of_life.h"
#include "kernels.cuh"
#include "utils.h"
#include <time.h>
#include <string.h>

void initializeGrid(unsigned char* grid, int width, int height, float density) {
    srand(time(NULL));
    int threshold = (int)(density * 100.0f);
    for (int i = 0; i < width * height; i++) {
        grid[i] = (rand() % 100 < threshold) ? 1 : 0;
    }
}

static void exchangeHalos(GPUContext* ctx, int gridWidth) {
    NCCL_CHECK(ncclGroupStart());
    
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        
        int prev = (i - 1 + ctx->nDevices) % ctx->nDevices;
        int next = (i + 1) % ctx->nDevices;
        
        // Send bottom row to next GPU's top halo
        NCCL_CHECK(ncclSend(
            ctx->d_grid[i] + gridWidth * ctx->localHeight,
            gridWidth, ncclUint8, next, ctx->comms[i], ctx->streams[i]
        ));
        
        // Send top row to previous GPU's bottom halo
        NCCL_CHECK(ncclSend(
            ctx->d_grid[i] + gridWidth * ctx->haloSize,
            gridWidth, ncclUint8, prev, ctx->comms[i], ctx->streams[i]
        ));
        
        // Receive from previous GPU into top halo
        NCCL_CHECK(ncclRecv(
            ctx->d_grid[i], gridWidth, ncclUint8, 
            prev, ctx->comms[i], ctx->streams[i]
        ));
        
        // Receive from next GPU into bottom halo
        NCCL_CHECK(ncclRecv(
            ctx->d_grid[i] + gridWidth * (ctx->localHeight + ctx->haloSize),
            gridWidth, ncclUint8, next, ctx->comms[i], ctx->streams[i]
        ));
    }
    
    NCCL_CHECK(ncclGroupEnd());
}

void runSimulation(
    GPUContext* ctx, 
    GridConfig config,
    unsigned char* h_initialGrid,
    int numIterations
) {
    // Copy initial data to GPUs
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(
            ctx->d_grid[i] + config.width * ctx->haloSize,
            h_initialGrid + i * ctx->localHeight * config.width,
            config.width * ctx->localHeight,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Setup kernel launch parameters
    dim3 blockDim(config.blockSize, config.blockSize);
    dim3 gridDim(
        (config.width + config.blockSize - 1) / config.blockSize,
        (ctx->localHeight + config.blockSize - 1) / config.blockSize
    );
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    printf("Starting simulation for %d iterations...\n", numIterations);
    
    // Main simulation loop
    for (int iter = 0; iter < numIterations; iter++) {
        // Exchange boundary rows
        exchangeHalos(ctx, config.width);
        
        // Launch kernel on each GPU
        for (int i = 0; i < ctx->nDevices; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            gameOfLifeKernel<<<gridDim, blockDim, 0, ctx->streams[i]>>>(
                ctx->d_grid[i] + config.width * ctx->haloSize,
                ctx->d_grid_next[i] + config.width * ctx->haloSize,
                config.width, ctx->localHeight, i * ctx->localHeight
            );
        }
        
        // Swap buffers
        for (int i = 0; i < ctx->nDevices; i++) {
            unsigned char* tmp = ctx->d_grid[i];
            ctx->d_grid[i] = ctx->d_grid_next[i];
            ctx->d_grid_next[i] = tmp;
        }
    }
    
    // Synchronize and measure time
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(ctx->streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Simulation completed!\n");
    printf("Total time: %.2f ms\n", milliseconds);
    printf("Average time per iteration: %.2f ms\n", milliseconds / numIterations);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void copyResultsToHost(GPUContext* ctx, GridConfig config, unsigned char* h_grid) {
    for (int i = 0; i < ctx->nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(
            h_grid + i * ctx->localHeight * config.width,
            ctx->d_grid[i] + config.width * ctx->haloSize,
            config.width * ctx->localHeight,
            cudaMemcpyDeviceToHost
        ));
    }
}