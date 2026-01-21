# silu_triton_test.py
# CSE 554 Group 14
# Testing and benchmarking of the SiLU Triton kernel.

import torch
from silu_triton_kernel import silu_triton
from silu_torch import silu as silu_pytorch

import numpy as np

def correctness(term_width: int = 80):
    """
    Test that the Triton kernel produces correct results.

    Args:
        term_width: Width of the terminal for formatting output.

    Returns:
        None
    """
    print("=" * term_width)
    print("Correctness Test")
    print("=" * term_width)

    print(f"{'Tensor Shape':<20} {'Test Status':<20} {'Max Diff':<20}")
    print("-" * term_width)

    # Test various tensor sizes
    test_sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192)
    ]

    for size in test_sizes:
        x = torch.randn(size, device="cuda", dtype=torch.float32)

        # Compute with PyTorch reference
        expected = silu_pytorch(x)

        # Compute with Triton kernel
        result = silu_triton(x)

        # Check correctness
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(result - expected)).item()
        status = "FAIL ❌"
        if torch.allclose(result, expected, rtol=1e-5, atol=1e-5):
            status = "PASS ✅"

        print(f"{str(size):<20} {status:<20} {max_diff:<20.2e}")

    print()

def benchmark_kernel(fn, x, num_warmup: int = 10, num_runs: int = 100):
    """
    Benchmark a kernel function.

    Args:
        fn: The kernel function to benchmark.
        x: Input tensor.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        runtime: Average runtime in milliseconds.
        bandwidth: Average effective bandwidth in GB/s.
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = fn(x)
    torch.cuda.synchronize()

    runtime, bandwidth = [], [] # Store summary statistics

    # Timed runs
    for i in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = fn(x)
        end.record()

        torch.cuda.synchronize()

        runtime.append(start.elapsed_time(end)) # milliseconds
        bandwidth.append(2 * x.element_size() * x.numel() / (runtime[-1] / 1000) / 1e9) # GB/s

    # Calculate summary statistics
    return (np.average(runtime), np.average(bandwidth))

def benchmark(term_width: int = 80):
    """
    Compare performance of Triton kernel vs PyTorch. Sends reults to
    STDOUT.

    Args:
        term_width: Width of the terminal for formatting output.

    Returns:
        None
    """

    # Test with different sizes
    test_sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    stdout_runtime, stdout_bandwidth = "", ""
    for size in test_sizes:
        x = torch.randn(size, device="cuda")

        # Benchmark respective implementations
        pytorch_time, pytorch_bandwidth = benchmark_kernel(silu_pytorch, x)
        triton_time, triton_bandwidth = benchmark_kernel(silu_triton, x)

        # Save runtime results
        speedup = pytorch_time / triton_time # runtime speed up
        stdout_runtime += f"{str(size):<20} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}\n"

        # Save bandwidth results
        speedup = triton_bandwidth / pytorch_bandwidth # bandwidth speed up
        stdout_bandwidth += f"{str(size):<20} {pytorch_bandwidth:<15.2f} {triton_bandwidth:<15.2f} {speedup:<10.2f}\n"

    # Print runtime benchmark results
    print("=" * term_width)
    print("Runtime Benchmark")
    print("=" * term_width)

    print(f"{'Size':<20} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * term_width)
    print(stdout_runtime)

    print()

    # Print bandwidth benchmark results
    print("=" * term_width)
    print("Bandwidth Benchmark")
    print("=" * term_width)

    print(f"{'Size':<20} {'PyTorch (GB/s)':<15} {'Triton (GB/s)':<15} {'Speedup':<10}")
    print("-" * term_width)
    print(stdout_bandwidth)

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}\n")

    torch.manual_seed(0) # Reproducibility
    term_width = 80 # max width of terminal output

    # Correctness separately, serves as a warmup
    correctness(term_width=term_width)

    # Calculate timing and bandwidth
    benchmark(term_width=term_width)
