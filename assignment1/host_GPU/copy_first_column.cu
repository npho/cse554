/**
 * copy_first_column.cu
 * CSE 554 Group 14
 * Efficient host-to-GPU transfer of a strided column from a row-major matrix.
 */

#include "copy_first_column.h"

#include <cuda_runtime.h>

/**
 * The idea is to first copy the strided column into a contiguous
 * buffer on the host, then transfer that buffer to the GPU.
 *
 * Pinned memory also introduces additional overhead that slows down
 * less compute intensive operations so we use pageable memory.
 */
void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    void* h_pageable = malloc(rows * sizeof(float));

    // Coalesce into contiguous buffer before copy
    for (int i = 0; i < rows; i++) {
        ((float*)h_pageable)[i] = h_A[i * cols];
    }

    // Send to GPU and clean up before exit
    cudaMemcpy(d_A, h_pageable, rows * sizeof(float), cudaMemcpyHostToDevice);
    free(h_pageable);
}
