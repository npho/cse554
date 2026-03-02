/**
 * prof_example.cu
 * CSE 554 Group 14
 * Code for profiling cuBLAS GEMM performance.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define BLOCKSIZE 256

using namespace std;

/**
 * GEMM for C = A * B in column-major order.
 *
 * Arguments:
 *  A: pointer to M x K matrix
 *  B: pointer to K x N matrix
 *  C: pointer to M x N matrix (output)
 *  m: number of rows in A and C
 *  n: number of columns in B and C
 *  k: number of columns in A and rows in B
 */
__global__ void GEMM(const float* A, const float* B, float* C, int m, int n, int k) {
    // In column-major GEMM, 'x' usually maps to rows and 'y' to columns
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float sum = 0.0f;
        for (int x = 0; x < k; x++) {
            sum += A[x * m + i] * B[j * k + x]; // A is (M x K), B is (K x N)
        }
        C[j * m + i] = sum; // C is (M x N)
    }
}

int main(int argc, char* argv[]) {
    // Parsing CLI args.
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <m> <n> <k>" << endl;
        return 1;
    }

    // I think of it as m, k, n but matching cuBLAS API m, n, k ordering for consistency.
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    float* host_A   = new float[m * k];
    float* host_B   = new float[k * n];
    float* host_C_1 = new float[m * n]; // CPU GEMM result
    float* host_C_2 = new float[m * n]; // cublasGemmEx result

    // Initialize host arrays
    for (int i = 0; i < (m*k); i++) {
        host_A[i] = i;
    }

    for (int i = 0; i < (k*n); i++) {
        host_B[i] = i;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, host_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Probably didn't need to do this part but verifying future cuBLAS result with a reference.
    dim3 num_threads(16, 16);
    dim3 num_block((n + 15) / 16, (m + 15) / 16);

    GEMM<<<num_block, num_threads>>>(d_A, d_B, d_C, m, n, k); // GPU GEMM reference result
    cudaDeviceSynchronize();
    cudaMemcpy(host_C_1, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    int warm_up_count = 100;
    int profile_count = 100;
    size_t L2_size = 50 * 1024 * 1024;

    for (int i = 0; i < warm_up_count; ++i)
    {
        // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     d_A, CUDA_R_32F, m,
                     d_B, CUDA_R_32F, k,
                     &beta,
                     d_C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    int* clear_l2_buffer;
    cudaMalloc(&clear_l2_buffer, L2_size);

    float total_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < profile_count; ++i)
    {
        cudaMemset(clear_l2_buffer, 0, L2_size); // Clear L2 cache https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh
        cudaEventRecord(start);
        // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     d_A, CUDA_R_32F, m,
                     d_B, CUDA_R_32F, k,
                     &beta,
                     d_C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float average_time = total_ms / profile_count;
    //std::cout << "Average time (ms): " << average_time << std::endl;
    std::cout << average_time << std::endl;

    // Copy GPU result back and verify result
    cudaMemcpy(host_C_2, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify cuBLAS result against GEMM reference
    for (int i = 0; i < (m*n); i++) {
        float diff = fabsf(host_C_1[i] - host_C_2[i]);
        float scale = fabsf(host_C_1[i]) + 1e-6f;
        if (diff / scale > 1e-3f) {
            std::cerr << "i=" << i << ":"
                      << " myGEMM=" << host_C_1[i]
                      << " cuBLAS=" << host_C_2[i]
                      << std::endl;
            break;
        }
    }

    // Free the L2 buffer
    cudaFree(clear_l2_buffer);
    // Free CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] host_A;
    delete[] host_B;
    delete[] host_C_1;
    delete[] host_C_2;

    return 0;
}
