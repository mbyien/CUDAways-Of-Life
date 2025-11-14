#include "kernels.cuh"

__global__ void gameOfLifeKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, 
    int localHeight, 
    int globalYOffset
) {
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