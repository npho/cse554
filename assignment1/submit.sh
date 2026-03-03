#!/bin/bash

assignment1=(
	# Section 1
    "silu/silu_torch.py"
	"silu/silu_triton_kernel.py"
	"silu/silu_triton_test.py"
	"silu/torch_silu.json"
	"silu/torch_silu.nsys-rep"
	"silu/CUDA/silu.cu"
	"silu/CUDA/main.cu"
	"silu/CUDA/cuda_silu.ncu-rep"

	# Section 2
	"rms_norm/matrix/rms_norm_matrix.cu"
	"rms_norm/matrix/main.cu"
	"rms_norm/matrix/rms_norm_matrix.ncu-rep"
	"rms_norm/vector/rms_norm_vector.cu"
	"rms_norm/vector/main.cu"
	"rms_norm/vector/rms_norm_vector.ncu-rep"

	# Section 3
	"host_GPU/main.cu"
	"host_GPU/copy_first_column.cu"
	"host_GPU/copy.ncu-rep"

	# Report
	#"hw1-group14.pdf" # Submitted separately
)

echo "Packing up assignment 1..."

# The -r flag makes it recursive (useful for directories)
zip -r assignment1-group14.zip "${assignment1[@]}"
