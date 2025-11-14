# CUDAways Of Life
## Features:

* Supports any number of GPUs (automatically detected)
* 1024x1024 grid with 30% initial cell density
* Horizontal wrapping (periodic boundary in x-direction)
* Vertical distribution with halo exchange via NCCL
* 1000 iterations by default
## Key Components:

**Domain Decomposition:** The grid is split horizontally across multiple GPUs, with each GPU handling a portion of the rows

**Halo Exchange:** Uses NCCL to exchange boundary rows between adjacent GPUs. It does that by having each GPU sends its top and bottom boundary rows to neighbors. Receives halo data from adjacent GPUs to compute boundary cells correctly


**NCCL Communication:** Uses ncclGroupStart/End for efficient collective operations with ring topology for periodic boundary conditions
*Performance:* Includes timing measurements to benchmark multi-GPU performance

## Compile and Run:
```cmake
nvcc -o game_of_life game_of_life.cu -lnccl
./game_of_life
```

## Next step
* Finish writing tests
* Benchmarking
* Create script to visualize results
* documentation on benchmarks and images generated
* generate extra layers so it's not 1024x1024, but 1024x1024x5
