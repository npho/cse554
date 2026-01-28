#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256

// The last argument is the size of the arrays
__global__ void add(int *a, int *b, int *c, int n) {
    // we need to calculate the start of each thread (since we control at the thread level)

    // blockIdx in CUDA is a built-in variable that contains the block index in the grid
    // blockIdx.x is the first dimension
    int block_start = blockIdx.x * blockDim.x;
    int thread_start = threadIdx.x;
    int i = block_start + thread_start;
    if (i < n) { // this is akin to the mask in Triton
        c[i] = a[i] + b[i];
    }
    // Chck out the addition diagram in the docs
    // Shouldn't we have some syncronization here?
}

int main(){
    size_t num = 100000000;
    int *a, *b, *c; // host pointers
    int *d_a, *d_b, *d_c; // device pointers: syntax is same, use naming to distinguish

    size_t size = num * sizeof(int);
    // Allocate memory on the host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Initialize input arrays
    for (size_t i = 0; i < num; i++) {
        a[i] = i;
        b[i] = i;
    }
    // Copy input arrays from host to device: this is the synchronized version
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    
    dim3 grid((num + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 block(BLOCKSIZE);
    // Launch kernel
    add<<<grid, block>>>(d_a, d_b, d_c, num);
    // Copy result array from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); // this is a blocking call so we don't need to sync after kernel launch (cudaDeviceSynchronize)

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Verify the result
    for (size_t i = 0; i < num; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Error at index " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            break;
        }
    }
    std::cout << "All results are correct!" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host memory
    free(a);
    free(b);
    free(c);
    return 0;
}