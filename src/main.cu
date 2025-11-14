#include "game_of_life.h"
#include "gpu_manager.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Configuration
    GridConfig config = {
        .width = 1024,
        .height = 1024,
        .blockSize = 16
    };
    int numIterations = 1000;
    float initialDensity = 0.3f;  // 30% alive
    
    // Parse command line arguments (optional)
    if (argc >= 2) numIterations = atoi(argv[1]);
    if (argc >= 3) config.width = atoi(argv[2]);
    if (argc >= 4) config.height = atoi(argv[3]);
    
    printf("=== Conway's Game of Life - Multi-GPU ===\n");
    printf("Grid size: %d x %d\n", config.width, config.height);
    printf("Iterations: %d\n", numIterations);
    printf("Initial density: %.1f%%\n", initialDensity * 100);
    
    // Initialize GPU context
    GPUContext* ctx = initGPUContext(config.height);
    if (!ctx) return 1;
    
    printf("Running on %d GPU(s)\n", ctx->nDevices);
    printf("Rows per GPU: %d\n\n", ctx->localHeight);
    
    // Allocate and initialize host grid
    unsigned char* h_grid = (unsigned char*)malloc(config.width * config.height);
    initializeGrid(h_grid, config.width, config.height, initialDensity);
    
    // Allocate device memory
    allocateGrids(ctx, config.width);
    
    // Run simulation
    runSimulation(ctx, config, h_grid, numIterations);
    
    // Optional: Copy results back
    // copyResultsToHost(ctx, config, h_grid);
    
    // Cleanup
    freeGrids(ctx);
    destroyGPUContext(ctx);
    free(h_grid);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}