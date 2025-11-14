#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macros
#define NCCL_CHECK(cmd) do {                          \
  ncclResult_t r = cmd;                               \
  if (r != ncclSuccess) {                             \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",        \
        __FILE__,__LINE__,ncclGetErrorString(r));     \
    exit(EXIT_FAILURE);                               \
  }                                                   \
} while(0)

#define CUDA_CHECK(cmd) do {                          \
  cudaError_t e = cmd;                                \
  if( e != cudaSuccess ) {                            \
    fprintf(stderr, "CUDA error %s:%d '%s'\n",        \
        __FILE__,__LINE__,cudaGetErrorString(e));     \
    exit(EXIT_FAILURE);                               \
  }                                                   \
} while(0)

#endif // UTILS_H