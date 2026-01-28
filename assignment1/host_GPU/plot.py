# plot.py
# CSE 554 Group 14
# Plots memory transfer bandwidth from benchmark results.

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bandwidth(csv_file: str, output_file: str) -> None:
    """
    Plot Host-to-GPU and GPU-to-Host bandwidth vs transfer size
    for both pageable and pinned memory.

    Args:
        csv_file: Path to CSV file with benchmark results.
        output_file: Path to save the plot (PNG format).

    Returns:
        None
    """
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Pageable memory
    if {'h2d_pageable_gbps', 'd2h_pageable_gbps'}.issubset(df.columns):
        ax.semilogx(
            df["bytes"],
            df["h2d_pageable_gbps"],
            "o--",
            label="H2D Pageable",
            linewidth=2,
            markersize=5,
            color="tab:red",
        )
        ax.semilogx(
            df["bytes"],
            df["d2h_pageable_gbps"],
            "s--",
            label="D2H Pageable",
            linewidth=2,
            markersize=5,
            color="tab:green",
        )

    # Pinned memory
    if {'h2d_pinned_gbps', 'd2h_pinned_gbps'}.issubset(df.columns):
        ax.semilogx(
            df["bytes"],
            df["h2d_pinned_gbps"],
            "o-",
            label="H2D Pinned",
            linewidth=2,
            markersize=6,
            color="tab:blue",
        )
        ax.semilogx(
            df["bytes"],
            df["d2h_pinned_gbps"],
            "s-",
            label="D2H Pinned",
            linewidth=2,
            markersize=6,
            color="tab:orange",
        )

    ax.set_xlabel("Transfer Size (Bytes)", fontsize=12)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_title("PCIe Memory Transfer Bandwidth", fontsize=14)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    # Set x-axis ticks at powers of 2
    powers = np.arange(0, 21)
    tick_positions = 2**powers
    tick_labels = [f"$2^{{{p}}}$" for p in powers]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10, rotation=0)
    ax.tick_params(axis="both", which="both", direction="in")

    ax.set_xlim(0.5, 2**21)
    ax.set_ylim(0, 16)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSE 554 Group 14 Plot memory transfer bandwidth from benchmark results."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="benchmark.csv",
        help="Path to CSV file with benchmark results (default: benchmark.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="bandwidth.png",
        help="Output file path for the plot (e.g., bandwidth.png).",
    )

    args = parser.parse_args()
    plot_bandwidth(args.csv_file, args.output)
