#!/bin/bash

assignment4=(
	# Section 1
    Section1/*.py

	# Section 2
	Section2/*.py

	# Report
	#"hw4-group14.pdf" # Submitted separately
)

echo "Packing up assignment 4..."

# The -r flag makes it recursive (useful for directories)
zip -r assignment4-group14.zip "${assignment4[@]}"
