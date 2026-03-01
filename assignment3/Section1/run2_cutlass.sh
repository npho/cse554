#!/bin/bash

# run2_cutlass.sh
# CSE 554 Group 14
# Benchmarking various matrix sizes for CUTLASS SGEMM performance.

PROFILER="./cutlass-4.3.5/cmake/tools/profiler/cutlass_profiler"
OUTPUT="gemm_perf.csv"
LIBRARY="cutlass"

#declare -a NK_PAIRS=("512 512" "4096 4096" "14336 4096" "4096 1024" "1024 4096")
declare -a NK_PAIRS=("1024 4096")

# Write header only if the file does not already exist
if [ ! -f "$OUTPUT" ]; then
    echo "batch_size,N,K,library,tflops" > "$OUTPUT"
fi

for M in $(seq 128 128 2048); do
    for nk in "${NK_PAIRS[@]}"; do
        read -r N K <<< "$nk"

        output=$("$PROFILER" --seed=0 --profiling-iterations=100 --split_k_slices=1,2,4,8 --split_k_mode=serial --kernels=sgemm --m="$M" --n="$N" --k="$K" 2>/dev/null)

        # Extract GFLOPs from output
        gflops=$(echo "$output" | grep ^"[1-4]" | awk -F, '{print $NF}' | sort -gr | head -n 1)

        if [ -z "$gflops" ]; then
            echo "WARNING: no output for M=$M N=$N K=$K, skipping" >&2
            continue
        fi

        # Convert GFLOP/s -> TFLOP/s
        tflops=$(awk "BEGIN { printf \"%.6f\", $gflops / 1000 }")
        echo "$M,$N,$K,$LIBRARY,$tflops" >> "$OUTPUT"
        echo -e "M=$M\tN=$N\tK=$K\tTFLOPs=${tflops}"
    done
done

echo ""
echo "[*] Results appended to $OUTPUT"
python3 plot_gemm.py hw3-s1-q2b.png
echo "[*] Plotting results to hw3-s1-q2b.png"
