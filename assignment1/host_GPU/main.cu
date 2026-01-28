/**
 * main.cu
 * CSE 554 Group 14
 * Benchmark Host-to-GPU and GPU-to-Host memory transfer bandwidth.
 */

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

#include "copy_first_column.h"

struct BenchmarkResult {
    size_t bytes;
    double h2d_pageable_GBps;
    double d2h_pageable_GBps;
    double h2d_pinned_GBps;
    double d2h_pinned_GBps;
};

double HostDeviceCopy(void* src, void* dst, size_t bytes, int n, cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    cudaMemcpy(dst, src, bytes, kind);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < n; i++) {
        cudaMemcpy(dst, src, bytes, kind);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate statistics
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds = (ms / 1000.0) / n; // average time over n iterations
    const double giga = 1e9;
    double gbps = (static_cast<double>(bytes) / giga) / seconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gbps;
}

BenchmarkResult TestTransfers(size_t bytes, int n = 100) {
    BenchmarkResult result;
    result.bytes = bytes;

    // Allocate GPU device memory
    void* d_data;
    cudaMalloc(&d_data, bytes);

    // Pageable (non-pinned) memory tests
    void* h_pageable = malloc(bytes);
    memset(h_pageable, 0xAB, bytes);

    result.h2d_pageable_GBps = HostDeviceCopy(h_pageable, d_data, bytes, n, cudaMemcpyHostToDevice);
    result.d2h_pageable_GBps = HostDeviceCopy(d_data, h_pageable, bytes, n, cudaMemcpyDeviceToHost);

    free(h_pageable);

    // Pinned memory tests
    void* h_pinned;
    cudaMallocHost(&h_pinned, bytes);
    memset(h_pinned, 0xAB, bytes);

    result.h2d_pinned_GBps = HostDeviceCopy(h_pinned, d_data, bytes, n, cudaMemcpyHostToDevice);
    result.d2h_pinned_GBps = HostDeviceCopy(d_data, h_pinned, bytes, n, cudaMemcpyDeviceToHost);

    cudaFreeHost(h_pinned);

    cudaFree(d_data);

    return result;
}

int main(int argc, char* argv[]) {
    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << std::endl;

    // Determine output filename
    const char* output_file = "benchmark.csv";
    if (argc > 1) {
        output_file = argv[1];
    }

    const int term_width = 80;

    // Print header
    std::cout << std::right << std::setw(8) << "Bytes"
              << std::right << std::setw(12) << "H2D Page"
              << std::right << std::setw(12) << "D2H Page"
              << std::right << std::setw(12) << "H2D Pinned"
              << std::right << std::setw(12) << "D2H Pinned"
              << std::right << std::setw(12) << "H2D Speedup"
              << std::right << std::setw(12) << "D2H Speedup"
              << std::endl;
    std::cout << std::string(term_width, '-') << std::endl;

    // Measure transfers for 2^0 to 2^20 bytes
    std::vector<BenchmarkResult> results;
    std::cout << std::fixed << std::setprecision(3);
    for (int exp = 0; exp <= 20; exp++) {
        size_t bytes = static_cast<size_t>(1) << exp;

        // More iterations for small transfers to get stable averages
        int iterations = (bytes < 1024) ? 1000 : 100;

        BenchmarkResult result = TestTransfers(bytes, iterations);
        results.push_back(result);

        const double h2d_speedup = result.h2d_pinned_GBps / result.h2d_pageable_GBps;
        const double d2h_speedup = result.d2h_pinned_GBps / result.d2h_pageable_GBps;
        std::cout << std::right << std::setw(8) << bytes
                  << std::right << std::setw(12) << result.h2d_pageable_GBps
                  << std::right << std::setw(12) << result.d2h_pageable_GBps
                  << std::right << std::setw(12) << result.h2d_pinned_GBps
                  << std::right << std::setw(12) << result.d2h_pinned_GBps
                  << std::right << std::setw(12) << h2d_speedup
                  << std::right << std::setw(12) << d2h_speedup
                  << std::endl;
    }

    // Write results to CSV
    std::ofstream csv(output_file);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return EXIT_FAILURE;
    }

    csv << "bytes,h2d_pageable_gbps,d2h_pageable_gbps,h2d_pinned_gbps,d2h_pinned_gbps" << std::endl;
    for (const auto& r : results) {
        csv << r.bytes << ","
            << r.h2d_pageable_GBps << ","
            << r.d2h_pageable_GBps << ","
            << r.h2d_pinned_GBps << ","
            << r.d2h_pinned_GBps << std::endl;
    }
    csv.close();

    std::cout << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;
    std::cout << std::endl;

    //
    // Benchmark copy_first_column
    //
    std::cout << std::string(term_width, '=') << std::endl;
    std::cout << "Strided Column Copy Benchmark (copy_first_column)" << std::endl;
    std::cout << std::string(term_width, '=') << std::endl;

    const int ROWS = 8192;
    const int COLS = 65536;
    const size_t matrix_bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);
    const size_t column_bytes = ROWS * sizeof(float);

    std::cout << "      Matrix: " << ROWS << " x " << COLS << " floats ("
              << (matrix_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "      Column: " << ROWS << " floats (" << (column_bytes / 1024.0) << " KB)" << std::endl;

    // Allocate host matrix pageable memory
    float* h_matrix = static_cast<float*>(malloc(matrix_bytes));
    if (!h_matrix) {
        std::cerr << "Failed to allocate host matrix" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize the random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> U(-10.0f, 10.0f); // [-10, 10]

    // Initialize matrix with known random values
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            h_matrix[r * COLS + c] = U(gen);
        }
    }

    // Allocate device column buffer
    float* d_column;
    cudaMalloc(&d_column, column_bytes);

    // Warm-up call (first call allocates pinned staging buffer)
    copy_first_column(h_matrix, d_column, ROWS, COLS);
    cudaDeviceSynchronize();

    // Benchmark with CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int NUM_ITERATIONS = 100;

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        copy_first_column(h_matrix, d_column, ROWS, COLS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Verify speed
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_us = (total_ms * 1000.0) / NUM_ITERATIONS;
    const char* test_time = avg_us < 100.0 ? "PASS" : "FAIL";

    // Verify correctness, copy result back and check values w/in tolerance
    std::vector<float> h_result(ROWS);
    cudaMemcpy(h_result.data(), d_column, column_bytes, cudaMemcpyDeviceToHost);

    float error = 0.0f;
    for (int r = 0; r < ROWS; r++) {
        int idx = r * COLS; // First column
        error += fabsf(h_matrix[idx] - h_result[r]);
    }

    bool test_acc = error < 1e-3f ? true : false;

    std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_us << " μs (" << test_time << ")" << std::endl;
    std::cout << "    Accuracy: " << error << " (" << (test_acc ? "PASS" : "FAIL") << ")\n" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_column);
    free(h_matrix);

    return 0;
}
