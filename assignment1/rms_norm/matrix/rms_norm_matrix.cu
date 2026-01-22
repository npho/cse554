#include <cuda_runtime.h>
#include <stdio.h>

// Number of threads per block for Quadro RTX 6000
#define THREADS_PER_BLOCK 256

__global__ void rms_norm_matrix_kernel(
    float *input,
    float *weight,
    float *output,
    int rows,
    int cols,
    float epsilon
) {
    extern __shared__ float shared_mem[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row < rows) {
        // Phase 1: Parallel reduction for sum of squares
        float thread_sum = 0.0f;
        for (int j = tid; j < cols; j += blockSize) {
            float val = input[row * cols + j];
            thread_sum += val * val;
        }
        shared_mem[tid] = thread_sum;
        __syncthreads();
        
        // Reduction in shared memory
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        
        float rms = rsqrtf(shared_mem[0] / cols + epsilon);
        
        // Phase 2: Normalize and apply weight (coalesced)
        for (int j = tid; j < cols; j += blockSize) {
            float normalized = input[row * cols + j] * rms;
            output[row * cols + j] = normalized * weight[j];
        }
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = THREADS_PER_BLOCK;
    size_t shared_mem_size = block_size * sizeof(float);
    rms_norm_matrix_kernel<<<rows, block_size, shared_mem_size>>>(d_input, d_weight, d_output, rows, cols, epsilon);

    // Copy result back to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
