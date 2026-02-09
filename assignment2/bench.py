# bench.py

import argparse
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from no_kv import Engine as NoKV
from single_batch import Engine as SingleBatch
from uniform_prefill import Engine as UniformPrefill
from different_prefill import Engine as DifferentPrefill

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmarking KV cache implementations.")

	### Q1 ###
	parser.add_argument("-c1", "--csv1", type=str, default="q1.csv", help="Save Q1 benchmark results as CSV file for future plotting.")
	parser.add_argument("-p1", "--plot1", type=str, default="q1.png", help="Save Q1 benchmark results as plot image for future plotting.")

	### Q2 ###
	parser.add_argument("-c2", "--csv2", type=str, default="q2.csv", help="Save Q2 benchmark results as CSV file for future plotting.")
	parser.add_argument("-p2", "--plot2", type=str, default="q2.png", help="Save Q2 benchmark results as plot image for future plotting.")

	args = parser.parse_args()

	"""
	Q1: Fix the input length at 1024 tokens, and vary the output length from 128 to 2048 in steps of 128. Compare two implementations: the original implementation without KV cache, and your implementation with KV cache.
	"""
	input_string = "Hi, who are you?" * 170 # Approx 1024 tokens
	engines = [NoKV(), SingleBatch()]
	rounds = np.arange(128, 2048+1, 128)
	#rounds = [10, 20, 30] # DEBUG

	inputs, outputs, elapsed = [], [], {}
	for engine in engines:
		elapsed[engine.__name__] = []
		for i, r in enumerate(rounds):
			print(f"Running {engine.__name__} ... ", end="")
			t0 = time.perf_counter()
			input_ids, output_text = engine.generate(input_string, rounds=r)
			t1 = time.perf_counter()
			e = t1 - t0

			print(f"{len(input_ids)} in and {r} out took {e:.4f} seconds!")

			# Correctness check
			if engine.__name__ != "NoKV":
				assert output_text == outputs[i], "Output mismatch between NoKV and SingleBatch implementations!"
			else:
				inputs.append(len(input_ids))
				outputs.append(output_text)

			# Store results for later processing
			elapsed[engine.__name__].append(e)

	if args.csv1:
		with open(args.csv1, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Engine', 'Input', 'Rounds', 'Time'])
			for engine in engines:
				for i, r in enumerate(rounds):
					writer.writerow([engine.__name__, inputs[i], r, elapsed[engine.__name__][i]])

	"""
	Q2: Profile the end-to-end generation time (only consider the generation, exclude model loading and other overheads) with a batch size of 2**0 - 2**6. Each request has an input length of 512 and an output length of 128. Compute the throughput token/s for different batch sizes.
	"""
