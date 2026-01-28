# silu_torch.py
# CSE 554 Group 14
# Implementation and profiling of the SiLU activation function using PyTorch.

import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function aka Swish.

    Args:
        x: Input tensor.

    Returns:
        Tensor after applying SiLU activation.
    """
    return x / (1.0 + torch.exp(-x))

if __name__ == "__main__":
    # fancy command line arguments
    parser = argparse.ArgumentParser(description="CSE 554 Group 14 SiLU PyTorch implementation and performance profiling.")
    parser.add_argument("-s", "--size", type=int, default=8192, help="Size of the input tensor (default: 8192).")
    parser.add_argument("-w", "--warmup", type=int, default=100, help="Number of warmup iterations (default: 100).")
    parser.add_argument("-o", "--output", default="torch_silu.json", help="File name for PyTorch trace (default: torch_silu.json).")
    args = parser.parse_args()

    torch.manual_seed(0) # Reproducibility
    term_width = 80 # max width of terminal output

    print("=" * term_width)
    print("PyTorch SiLU implementation and performance profiling")
    print("=" * term_width)

    # Default to CPU if no CUDA.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[+] CUDA device: {torch.cuda.get_device_name(0)}")

    # Create random tensor
    x = torch.rand(args.size, args.size, device=device)
    print(f"[+] Tensor shape: {x.shape}")

    # Warmup the GPU
    print(f"[+] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        _ = silu(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    ) as prof:
        # Use record_function to mark the section of code to be profiled
        with record_function("silu"):
            # Timed runs
            start.record()
            result = silu(x)
            end.record()

    if device == "cuda":
        torch.cuda.synchronize()

    # export to chrome trace
    print(f"[+] Exporting trace to {args.output}")
    prof.export_chrome_trace(args.output)

    print("\n" + "=" * term_width)
    print("Performance Numbers")
    print("=" * term_width)
    func_time = start.elapsed_time(end) # msec

    print("[+] Time for SiLU:", func_time, "ms")
    print("[+] Bandwidth:", (2 * x.element_size() * x.numel() / 1e9) / (func_time / 1e3), "GBps")

    # compare with default PyTorch SiLU for correctness
    expected = torch.nn.functional.silu(x)
    max_diff = torch.max(torch.abs(result - expected)).item()

    status = "FAIL"
    if torch.allclose(result, expected, atol=1e-3):
        status = "PASS"

    print("\n" + "=" * term_width)
    print("Correctness Test")
    print("=" * term_width)

    print(f"[+] Test closeness: {status} (<{max_diff:.2e})\n")

    """
    print("=" * term_width)
    print("Torch Profiler Results")
    print("=" * term_width)

    # https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=10))
    """
