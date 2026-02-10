#!/bin/bash

assignment2=(
	# Section 2
    "single_batch.py"
	"uniform_prefill.py"
	"different_prefill.py"

	# Report
	#"hw2-group14.pdf" # Submitted separately
)

echo "Packing up assignment 2..."

# The -r flag makes it recursive (useful for directories)
zip -r assignment2-group14.zip "${assignment2[@]}"
