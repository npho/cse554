# intensity.py
# CSE 554 Group 14
# Code to generate LaTeX for the output for HW3 Q1B.

def intensity(M:int, K:int, N:int, S:int = 1) -> int:
	"""
	Calculate the intensity of a matrix multiplication operation.

	Parameters:
		- M: Number of rows in the first matrix.
		- K: Number of columns in the first matrix and rows in the second matrix.
		- N: Number of columns in the second matrix.
		- S: Size of each element in bytes (default is 1 byte).

	Returns:
		- Intensity: The ratio of total floating-point operations to total data movement in bytes.
	"""
	W = M*N*(2*K - 1) # Total floating-point operations
	Q = S*(M*K + K*N + M*N) # Total data movement in bytes
	I = W / Q # Intensity calculation
	return I

if __name__ == "__main__":
	NK = [
		(512, 512),
		(4096, 4096),
		(14336, 4096),
		(4096, 1024),
		(1024, 4096)
	]

	for M in range(128, 2048 + 1, 128):
		print(f"{M}", end="")
		for N, K in NK:
			#print(f"M={M}, K={K}, N={N}", end="")
			I = intensity(M, K, N)
			print(f" & {I:.2f}", end="")
		print(" \\\\")
