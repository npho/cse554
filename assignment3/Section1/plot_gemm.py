import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load CSV
# batch_size,N,K,library,tflops
df = pd.read_csv('gemm_perf.csv')

# Get all unique (N, K) shapes
shapes = df[['N', 'K']].drop_duplicates().values.tolist()

# Plot for each shape
for i, (N, K) in enumerate(shapes, 1):
    shape_df = df[(df['N'] == N) & (df['K'] == K)]
    batch_sizes = sorted(shape_df['batch_size'].unique())

    plt.figure(figsize=(8, 5))

    for lib in ['cutlass', 'cublas']:
        lib_df = shape_df[shape_df['library'] == lib]
        lib_df = lib_df.sort_values(by='batch_size')
        lib_df = lib_df.groupby('batch_size')['tflops'].max().reset_index()
        plt.plot(lib_df['batch_size'], lib_df['tflops'], marker='o', label=lib.capitalize())

    plt.title(f'Performance for Shape N={N}, K={K}')
    plt.xlabel('Batch Size')
    plt.ylabel('TFLOPs')
    plt.grid(True)
    plt.legend(["CUTLASS", "cuBLAS"])
    plt.tight_layout()
    #plt.show()
    out = f"hw3-s1-q2-{i}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[*] Saved plot for N={N}, K={K} as {out}")
