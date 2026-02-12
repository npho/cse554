# bench.py
# CSE 554 Group 14
# Code to benchmark KV cache and batched implementations.

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
	parser.add_argument("-q1", "--question1", action="store_true", default=False, help="Run benchmark for Question 1.")
	parser.add_argument("-c1", "--csv1", type=str, default="q1.csv", help="Save Q1 benchmark results as CSV file for future plotting.")
	parser.add_argument("-p1", "--plot1", type=str, default="q1.png", help="Save Q1 benchmark results as plot image for future plotting.")

	### Q2 ###
	parser.add_argument("-q2", "--question2", action="store_true", default=False, help="Run benchmark for Question 2.")
	parser.add_argument("-c2", "--csv2", type=str, default="q2.csv", help="Save Q2 benchmark results as CSV file for future plotting.")
	parser.add_argument("-p2", "--plot2", type=str, default="q2.png", help="Save Q2 benchmark results as plot image for future plotting.")

	args = parser.parse_args()

	"""
	Q1: Fix the input length at 1024 tokens, and vary the output length from 128 to 2048 in steps of 128. Compare two implementations: the original implementation without KV cache, and your implementation with KV cache.
	"""
	if args.question1:
		print("Running benchmark for Question 1 ...")
		input_string = "Hi " * 1023 # Actually 1024 tokens
		engines = [NoKV(), SingleBatch()]
		rounds = np.arange(128, 2048+1, 128)
		#rounds = [10, 20, 30] # DEBUG

		inputs, outputs, elapsed = [], [], {}
		for engine in engines:
			elapsed[engine.__name__] = []
			for i, r in enumerate(rounds):
				print(f"Running {engine.__name__} ... ", end="")
				t0 = time.perf_counter()
				input_ids, output_text = engine.bench(input_string, rounds=r)
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
	if args.question2:
		print("Running benchmark for Question 2 ...")
		batch_sizes = [2**i for i in range(7)]
		engines = [UniformPrefill(), DifferentPrefill()]
		input_string = "Hi " * 511 # Actually 512 tokens
		rounds = 128

		inputs, outputs, elapsed = [], [], {}
		for engine in engines:
			elapsed[engine.__name__] = []
			for i, batch_size in enumerate(batch_sizes):
				print(f"Running {engine.__name__} ... ", end="")
				batch_input = [input_string for _ in range(batch_size)]
				t0 = time.perf_counter()
				input_ids_list, output_text = engine.bench(batch_input, rounds=rounds)
				t1 = time.perf_counter()
				tokens = len(input_ids_list[0]) # Just correctness check first, they're all the same
				e = t1 - t0

				print(f"batch size {batch_size} of {tokens} tokens generating {rounds} tokens took {e:.4f} seconds!")
				# Correctness check
				if engine.__name__ != "UniformPrefill":
					assert output_text[0] == outputs[i], "Output mismatch between NoKV and UniformPrefill or DifferentPrefill implementations!"
				else:
					inputs.append(tokens)
					outputs.append(output_text[0])

				# Store results for later processing
				elapsed[engine.__name__].append(e)

		if args.csv2:
			with open(args.csv2, 'w', newline='') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(['Engine', 'Input', 'Batch', 'Rounds', 'Elapsed'])
				for engine in engines:
					for i, batch_size in enumerate(batch_sizes):
						writer.writerow([engine.__name__, inputs[i], batch_size, rounds, elapsed[engine.__name__][i]])
