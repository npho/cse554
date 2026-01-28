#include <cuda_runtime.h>
#include <stdio.h>

// 1- Shared memory tile size: 128x128*4 = 64KB > 48 KB
// We need to use a dynamic kernel launch, since the shared memory size we'd fit this tile is larger than the default static allocation.
// We need to adjust the kernel launch parameters to allocate sufficient dynamic shared memory.
#define TILE_SIZE 128

__global__ void transposeKernel(float* input, float* output, int num_rows, int num_cols) {
    extern __shared__ float tile[]; // 2- no need to specify size, it will be provided at kernel launch
    int row_start = blockIdx.y * TILE_SIZE;
    // each thread will read one element of a tile column
    int col_start = blockIdx.x * TILE_SIZE + threadIdx.x;

    // load data into shared memory
    for (int i = 0; i < TILE_SIZE; i += 1) { // for every row in the tile
        if (row_start + i >= num_rows || col_start >= num_cols) { // this check is for the last tile which may be partial, zero out elements outside the matrix
            tile[i * TILE_SIZE + threadIdx.x] = 0.0f;
        } else {
            // When writing, we rad the tile, a row at a time
            tile[i * TILE_SIZE + threadIdx.x] = input[(row_start + i) * num_cols + col_start];
        }
    }
    // We need to synchronize to make sure the tile is fully loaded (avoid race conditions)
    // In general if your threads are going to reuse shared memory, you'll need to synchronize
    __syncthreads();

    int output_row_start = blockIdx.x * TILE_SIZE;
    int output_col_start = blockIdx.y * TILE_SIZE + threadIdx.x;

    int num_output_rows = num_cols;
    int num_output_cols = num_rows;

    // write transposed data to output
    for (int i = 0; i < TILE_SIZE; i += 1) {
        if (output_row_start + i >= num_output_rows ||  output_col_start >= num_output_cols) {
            return;
        }
        // When reading the tile, we read a column at a time: you isolate the uncoalesced access to shared memory
        output[(output_row_start + i) * num_output_cols + output_col_start] = tile[threadIdx.x * TILE_SIZE + i];
    }
}

int main(){
    int num_rows = 8192;
    int num_cols = 8192;

    float *h_input, *h_output;
    float *d_input, *d_output;

    size_t size = num_rows * num_cols * sizeof(float);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Initialize input matrix
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            h_input[i * num_cols + j] = static_cast<float>(i * num_cols + j);
        }
    }

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 3- Adjusting kernel attributes for dynamic shared memory
    cudaFuncSetAttribute(transposeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(float) * TILE_SIZE * TILE_SIZE);

    // Launch kernel: the ideas is to partition the matrix into TILE_SIZE x TILE_SIZE tiles. Each block will handle one tile.
    dim3 gridDim((num_cols + TILE_SIZE - 1) / TILE_SIZE, (num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // A block will have TILE_SIZE threads to cover one row of the tile. Recall that's 4 warp (4x32). This is our first attempt.
    // 128 reads is needed to read in the whole tile
    dim3 blockDim(TILE_SIZE);
    // The third parameter is the dynamic shared memory size
    transposeKernel<<<gridDim, blockDim, sizeof(float) * TILE_SIZE * TILE_SIZE>>>(d_input, d_output, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy output matrix back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    // Verify the result
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (h_output[j * num_rows + i] != h_input[i * num_cols + j]) {
                printf("Mismatch at (%d, %d): %f != %f\n", i, j, h_output[j * num_rows + i], h_input[i * num_cols + j]);
                break;
            }
        }
    }
    printf("Transpose completed successfully.\n");
    
    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}