#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 128  // Use multiple blocks for better occupancy

__global__ void rms_norm_vector_kernel_sum(
    float *input,
    float *partial_sums,
    int cols
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int gridSize = blockSize * gridDim.x;
    
    // Each block processes a portion of the vector
    float thread_sum = 0.0f;
    for (int j = bid * blockSize + tid; j < cols; j += gridSize) {
        float val = input[j];
        thread_sum += val * val;
    }
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Reduction within block
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store partial result
    if (tid == 0) {
        partial_sums[bid] = shared_mem[0];
    }
}

__global__ void rms_norm_vector_kernel_normalize(
    float *input,
    float *weight,
    float *output,
    float rms,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int j = idx; j < cols; j += stride) {
        float normalized = input[j] * rms;
        output[j] = normalized * weight[j];
    }
}

void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    float *d_input, *d_weight, *d_output, *d_psums;
    cudaMalloc((void**)&d_input, cols * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, cols * sizeof(float));
    cudaMalloc((void**)&d_psums, NUM_BLOCKS * sizeof(float));

    cudaMemcpy(d_input, input, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Phase 1: Compute partial sums
    int sharedMemSize = THREADS_PER_BLOCK * sizeof(float);
    rms_norm_vector_kernel_sum<<<NUM_BLOCKS, THREADS_PER_BLOCK, sharedMemSize>>>(
        d_input, d_psums, cols);
    
    // Copy partial sums to host and compute final RMS
    float partial_sums[NUM_BLOCKS];
    cudaMemcpy(partial_sums, d_psums, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        total_sum += partial_sums[i];
    }
    float rms = rsqrtf(total_sum / static_cast<float>(cols) + epsilon);
    
    // Phase 2: Normalize with multiple blocks
    rms_norm_vector_kernel_normalize<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_input, d_weight, d_output, rms, cols);

    cudaMemcpy(output, d_output, cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_psums);
}
