/**
 * silu.cu
 * CSE 554 Group 14
 * CUDA implementation of the SiLU activation function.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void silu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] / (1.0f + __expf(-input[i]));
    }
}

void silu(float *input, float *output, int n) {
    // Calculate grid dimensions
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    silu_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, n);
}
