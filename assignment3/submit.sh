#!/bin/bash

assignment3=(
	# Section 1
    "section1/prof_example.cu"

	# Section 2
	# TODO

	# Section 3
	"section3/flashinfer_pipeline.py"

	# Report
	#"hw3-group14.pdf" # Submitted separately
)

echo "Packing up assignment 3..."

# The -r flag makes it recursive (useful for directories)
zip -r assignment3-group14.zip "${assignment3[@]}"
