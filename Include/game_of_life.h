#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include "gpu_manager.h"

typedef struct {
    int width;
    int height;
    int blockSize;
} GridConfig;

// Initialize grid with random values
void initializeGrid(unsigned char* grid, int width, int height, float density);

// Run simulation for specified iterations
void runSimulation(
    GPUContext* ctx, 
    GridConfig config,
    unsigned char* h_initialGrid,
    int numIterations
);

// Copy results back to host
void copyResultsToHost(
    GPUContext* ctx,
    GridConfig config,
    unsigned char* h_grid
);

#endif // GAME_OF_LIFE_H