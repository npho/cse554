# plot.py
# CSE 554 Group 14
# Plots KV cache and batched implementations.

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def q1(csv_file, plot_file):
    """
    Plot the benchmark results for Question 1: NoKV vs SingleBatch.

    Args:
        csv_file (str): Path to the CSV file containing benchmark results for Q1.
        plot_file (str): Path to save the generated plot image for Q1.

    Returns:
        None
    """
    df = pd.read_csv(csv_file)
    df["Tokens per Second"] = df["Input"] / df["Time"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    #ax2 = ax1.twinx()

    for engine, group in df.groupby("Engine"):
        ax1.plot(group["Rounds"], group["Time"], marker="o", label=f"{engine}")
        #ax2.plot(group["Rounds"], group["Tokens per Second"], marker="s", linestyle="--",
        #        label=f"{engine} (Tokens/s)")

    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Time (s)")
    ax1.grid(visible=True, linestyle="--", alpha=0.3)
    ax1.tick_params(axis="both", which="both", direction="in")
    #ax2.set_ylabel("Tokens per Second")

    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left")
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Generation Time: NoKV vs SingleBatch")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Q1 plot saved: {plot_file}")

    return None

def q2(csv_file, plot_file):
    """
    Plot the benchmark results for Question 2: Uniform Prefill vs Different Prefill.

    Args:
        csv_file (str): Path to the CSV file containing benchmark results for Q2.
        plot_file (str): Path to save the generated plot image for Q2.

    Returns:
        None
    """
    df = pd.read_csv(csv_file)
    df["Tokens per Second"] = df["Batch"] * df["Rounds"] / df["Elapsed"]

    df = df[df["Engine"] == "UniformPrefill"]

    """
    Plot Runtime
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for engine, group in df.groupby("Engine"):
        ax1.plot(group["Batch"], group["Elapsed"], marker="o", label=f"{engine}")

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (s)")
    ax1.grid(visible=True, linestyle="--", alpha=0.3)
    ax1.tick_params(axis="both", which="both", direction="in")

    #lines1, labels1 = ax1.get_legend_handles_labels()
    #ax1.legend(lines1, labels1, loc="upper left")

    plt.title("Batched Processing Time")
    plt.tight_layout()
    plt.savefig(plot_file.replace(".png", "-time.png"), dpi=300, bbox_inches="tight")
    print(f"Q2 time plot saved: {plot_file.replace('.png', '-time.png')}")

    """
    Plot Throughput (Tokens/s)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for engine, group in df.groupby("Engine"):
        ax1.plot(group["Batch"], group["Tokens per Second"], marker="s", linestyle="--", label=f"{engine}")

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Tokens per Second")
    ax1.grid(visible=True, linestyle="--", alpha=0.3)
    ax1.tick_params(axis="both", which="both", direction="in")

    #lines1, labels1 = ax1.get_legend_handles_labels()
    #ax1.legend(lines1, labels1, loc="upper left")

    plt.title("Batched Throughput")
    plt.tight_layout()
    plt.savefig(plot_file.replace(".png", "-throughput.png"), dpi=300, bbox_inches="tight")
    print(f"Q2 throughput plot saved: {plot_file.replace('.png', '-throughput.png')}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking KV cache implementations.")

    ### Q1 ###
    parser.add_argument("-c1", "--csv1", type=str, default="q1.csv", help="Load Q1 benchmark results from CSV file for plotting.")
    parser.add_argument("-p1", "--plot1", type=str, default="hw2-s2-q1.png", help="Plot Q1 benchmark to image file.")

    ### Q2 ###
    parser.add_argument("-c2", "--csv2", type=str, default="q2.csv", help="Load Q2 benchmark results from CSV file for plotting.")
    parser.add_argument("-p2", "--plot2", type=str, default="hw2-s2-q2.png", help="Plot Q2 benchmark to image file.")

    args = parser.parse_args()

    if args.csv1:
        q1(args.csv1, args.plot1)

    if args.csv2:
        q2(args.csv2, args.plot2)
