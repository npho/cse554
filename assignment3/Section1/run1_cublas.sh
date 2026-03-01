#!/bin/bash

# run1_cublas.sh
# CSE 554 Group 14
# Benchmarking various matrix sizes for cuBLAS SGEMM performance.

BINARY="./prof_example"
OUTPUT="gemm_perf.csv"
LIBRARY="cublas"

#declare -a NK_PAIRS=("512 512" "4096 4096" "14336 4096" "4096 1024" "1024 4096")
declare -a NK_PAIRS=("1024 4096")

echo "batch_size,N,K,library,tflops" > "$OUTPUT"

for M in $(seq 128 128 2048); do
    for nk in "${NK_PAIRS[@]}"; do
        read -r N K <<< "$nk"

        avg_ms=$("$BINARY" "$M" "$N" "$K" 2>/dev/null)

        if [ -z "$avg_ms" ]; then
            echo "WARNING: no output for M=$M N=$N K=$K, skipping" >&2
            continue
        fi

        tflops=$(awk "BEGIN { printf \"%.6f\", ($M * $N * (2*$K - 1)) / 1e12 / ($avg_ms / 1e3) }")
        echo "$M,$N,$K,$LIBRARY,$tflops" >> "$OUTPUT"
        echo -e "M=$M\tN=$N\tK=$K\tms=${avg_ms}\tTFLOPs=${tflops}"
    done
done

echo ""
echo "Results written to $OUTPUT"
#python3 plot_gemm.py
