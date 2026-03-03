#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>


int main() {
    static constexpr size_t N = 1024 * 1024;

    // Allocate host memory
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_weight = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float epsilon = 1e-6f;

    // Initialize input and weight
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < N; i++) {
        h_weight[i] = 1.0f; // Example weight initialization
    }

    // Call RMS normalization
    rms_norm_vector(h_input, h_weight, h_output, N, epsilon);

    // Print some output values for verification
    for (size_t i = 0; i < 10; i++) {
        printf("h_output[%zu] = %f\n", i, h_output[i]);
    }
}
