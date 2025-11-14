#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GRID_WIDTH 1024
#define GRID_HEIGHT 1024
#define BLOCK_SIZE 16
#define NUM_ITERATIONS 1000

#define NCCL_CHECK(cmd) do {                          \
  ncclResult_t r = cmd;                               \
  if (r != ncclSuccess) {                             \
    printf("NCCL error %s:%d '%s'\n",                 \
        __FILE__,__LINE__,ncclGetErrorString(r));     \
    exit(EXIT_FAILURE);                               \
  }                                                   \
} while(0)

#define CUDA_CHECK(cmd) do {                          \
  cudaError_t e = cmd;                                \
  if( e != cudaSuccess ) {                            \
    printf("CUDA error %s:%d '%s'\n",                 \
        __FILE__,__LINE__,cudaGetErrorString(e));     \
    exit(EXIT_FAILURE);                               \
  }                                                   \
} while(0)

// CUDA kernel for Conway's Game of Life
__global__ void gameOfLifeKernel(const unsigned char* input, unsigned char* output, 
                                  int width, int localHeight, int globalYOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= localHeight) return;
    
    int neighbors = 0;
    
    // Count living neighbors (with wraparound for x, but not for y at GPU boundaries)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width;  // Wrap horizontally
            int ny = y + dy;
            
            // Only count if within local bounds (boundary cells handled by halo exchange)
            if (ny >= 0 && ny < localHeight) {
                neighbors += input[ny * width + nx];
            }
        }
    }
    
    unsigned char current = input[y * width + x];
    
    // Conway's rules
    if (current == 1) {
        output[y * width + x] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        output[y * width + x] = (neighbors == 3) ? 1 : 0;
    }
}

void initializeGrid(unsigned char* grid, int width, int height) {
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        grid[i] = (rand() % 100 < 30) ? 1 : 0;  // 30% alive
    }
}

int main(int argc, char** argv) {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    
    if (nDevices < 1) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("Running Conway's Game of Life on %d GPU(s)\n", nDevices);
    printf("Grid size: %d x %d\n", GRID_WIDTH, GRID_HEIGHT);
    
    // Calculate local height for each GPU
    int localHeight = GRID_HEIGHT / nDevices;
    int haloSize = 1;  // We need 1 row of halo cells on each side
    int localHeightWithHalo = localHeight + 2 * haloSize;
    
    // Allocate host memory for initial grid
    unsigned char* h_grid = (unsigned char*)malloc(GRID_WIDTH * GRID_HEIGHT);
    initializeGrid(h_grid, GRID_WIDTH, GRID_HEIGHT);
    
    // Arrays for device pointers and streams
    unsigned char** d_grid = (unsigned char**)malloc(nDevices * sizeof(unsigned char*));
    unsigned char** d_grid_next = (unsigned char**)malloc(nDevices * sizeof(unsigned char*));
    cudaStream_t* streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));
    ncclComm_t* comms = (ncclComm_t*)malloc(nDevices * sizeof(ncclComm_t));
    
    // Initialize NCCL
    NCCL_CHECK(ncclCommInitAll(comms, nDevices, NULL));
    
    // Setup for each GPU
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        // Allocate device memory with halo regions
        CUDA_CHECK(cudaMalloc(&d_grid[i], GRID_WIDTH * localHeightWithHalo));
        CUDA_CHECK(cudaMalloc(&d_grid_next[i], GRID_WIDTH * localHeightWithHalo));
        
        // Copy initial data (skip first halo row)
        CUDA_CHECK(cudaMemcpy(d_grid[i] + GRID_WIDTH * haloSize, 
                              h_grid + i * localHeight * GRID_WIDTH,
                              GRID_WIDTH * localHeight,
                              cudaMemcpyHostToDevice));
    }
    
    // Main simulation loop
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((GRID_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (localHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Starting simulation for %d iterations...\n", NUM_ITERATIONS);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Exchange halo regions using NCCL
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < nDevices; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            
            int prev = (i - 1 + nDevices) % nDevices;
            int next = (i + 1) % nDevices;
            
            // Send bottom row to next GPU's top halo
            NCCL_CHECK(ncclSend(d_grid[i] + GRID_WIDTH * localHeight,
                               GRID_WIDTH, ncclUint8, next, comms[i], streams[i]));
            
            // Send top row to previous GPU's bottom halo
            NCCL_CHECK(ncclSend(d_grid[i] + GRID_WIDTH * haloSize,
                               GRID_WIDTH, ncclUint8, prev, comms[i], streams[i]));
            
            // Receive from previous GPU into top halo
            NCCL_CHECK(ncclRecv(d_grid[i], GRID_WIDTH, ncclUint8, 
                               prev, comms[i], streams[i]));
            
            // Receive from next GPU into bottom halo
            NCCL_CHECK(ncclRecv(d_grid[i] + GRID_WIDTH * (localHeight + haloSize),
                               GRID_WIDTH, ncclUint8, next, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
        
        // Launch kernel on each GPU
        for (int i = 0; i < nDevices; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            gameOfLifeKernel<<<gridDim, blockDim, 0, streams[i]>>>(
                d_grid[i] + GRID_WIDTH * haloSize,
                d_grid_next[i] + GRID_WIDTH * haloSize,
                GRID_WIDTH, localHeight, i * localHeight);
        }
        
        // Swap buffers
        for (int i = 0; i < nDevices; i++) {
            unsigned char* tmp = d_grid[i];
            d_grid[i] = d_grid_next[i];
            d_grid_next[i] = tmp;
        }
    }
    
    // Synchronize all devices
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Simulation completed!\n");
    printf("Total time: %.2f ms\n", milliseconds);
    printf("Average time per iteration: %.2f ms\n", milliseconds / NUM_ITERATIONS);
    
    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_grid[i]));
        CUDA_CHECK(cudaFree(d_grid_next[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    
    free(h_grid);
    free(d_grid);
    free(d_grid_next);
    free(streams);
    free(comms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}