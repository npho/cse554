#include<cuda_runtime.h>
#include<stdio.h>

// 1- In this case we define the block size to be 256 elements
#define BLOCK_SIZE 256

__global__ void reduction_kernel(int *input, int *output, int N) {
    // We will again use share memory to hold the partial sums
    // since 256 elements is small enough, we can use the static allocation
    __shared__ int sdata[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? input[i] : 0; // This is similar to a mask operation
    __syncthreads();

    // Stride doubles in each step (see the figure from the slides)
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Only threads with index multiple of 2*s do the work
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        // After each addition, we need to synchronize the threads, since
        // in the next iteration, some threads will reuse results from the previous iteration
        __syncthreads();
    }
    // The final output will be written by thread 0 of each block
    if (tid == 0) {
        // This is similar to an atomic addition in a CPU. It allows different blocks to
        // add to global memory without any race conditions.
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