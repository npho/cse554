#!/bin/bash

# profile.sh
# CSE 554 Group 14
# Runs all the scripts at once and plots the results.

echo "=== Running CUBLAS GEMM profiling ==="
./run1_cublas.sh

echo ""

echo "=== Running CUTLASS GEMM profiling ==="
./run2_cutlass.sh
