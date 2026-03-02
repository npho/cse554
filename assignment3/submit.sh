#!/bin/bash

assignment3=(
	# Section 1
    "Section1/prof_example.cu"

	# Section 2
	"Section2/plot_attention_p.py"
	"Section2/plot_attention_d.py"
	"Section2/op_intensity.py"

	# Section 3
	"Section3/flashinfer_pipeline.py"

	# Report
	#"hw3-group14.pdf" # Submitted separately
)

echo "Packing up assignment 3..."

# The -r flag makes it recursive (useful for directories)
zip -r assignment3-group14.zip "${assignment3[@]}"
