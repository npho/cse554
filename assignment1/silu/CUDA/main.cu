/**
 * main.cu
 * CSE 554 Group 14
 * Correctness and performance testing for CUDA SiLU kernel.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "silu.h"
#include <string>

int main(int argc, char* argv[]) {
    int term_width = 80;
    std::cout << std::string(term_width, '=') << std::endl;
    std::cout << "SiLU Kernel Tests\n";
    std::cout << std::string(term_width, '=') << std::endl;

    // Print device info
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA device:\t%s\n", prop.name);

    srand(0); // reproducibility

    int n = 8192 * 8192; // benchmark size

    // Allocate host memory
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output_cpu = (float*)malloc(n * sizeof(float));
    float* h_output_gpu = (float*)malloc(n * sizeof(float));

    if (!h_input || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize the random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> U(-10.0f, 10.0f); // [-10, 10]

    // Fill input with random data
    for (int i = 0; i < n; ++i) {
        h_input[i] = U(gen);
    }

    // Compute CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        h_output_cpu[i] = h_input[i] / (1.0f + expf(-h_input[i]));
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 1; ++i) {
        silu(d_input, d_output, n); // Launch kernel
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice); // Reset input
    }
    cudaDeviceSynchronize();

    // Single timed run
    cudaEventRecord(start);
    silu(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy results back to host
    cudaMemcpy(h_output_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate memory bandwidth
    double bandwidth = (2.0f * n * sizeof(float) * 1e-9f) / (gpu_time_ms * 1e-3f); // GB/s

    // Verify correctness within tolerance
    float max_error = 0.0f;

    for (int i = 0; i < n; i++) {
        float error = fabsf(h_output_cpu[i] - h_output_gpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    bool PASSED = max_error < 1e-3f ? true : false;

    printf("CPU time:       %.4f ms\n", cpu_time_ms);
    printf("GPU time:       %.4f ms\n", gpu_time_ms);
    printf("Speedup:        %.2fx\n", cpu_time_ms / gpu_time_ms);
    printf("Bandwidth:      %.2f GB/s\n", bandwidth);
    printf("Max error:      %.2e\n", max_error);
    printf("Status:         %s\n\n", PASSED ? "PASSED" : "FAILED");

    // Free all the things
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return PASSED;
}
