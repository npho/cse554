#include<cuda_runtime.h>
#include<stdio.h>

#define BLOCK_SIZE 256

__global__ void reduction_kernel(int *input, int *output, int N) {
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // Please read the code and derive the math below on a piece of paper
    // Recall that each iteration is a step in a binary tree reduction. Each step has a specific stride s.
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid; //once in the loop, all threads from tid=0 to blockDim.x/2s operate simultaneously
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    int N = 1024*1024*1024;
    int *d_input, *d_output;
    int *h_input, *h_output;

    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i % 11; // Example data
    }
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block = BLOCK_SIZE;
    reduction_kernel<<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("Sum: %d\n", *h_output);
    // Check result
    int expected_sum = 0;
    for (int i = 0; i < N; i++) {
        expected_sum += h_input[i];
    }
    if (*h_output == expected_sum) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect! Expected %d, got %d\n", expected_sum, *h_output);
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}