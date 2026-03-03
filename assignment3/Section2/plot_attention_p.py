import flashinfer
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reference config for each model
# ---------------------------------------------------------------------------
llama3_1b_config = {
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

llama3_3b_config = {
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_key_value_heads": 8
}

llama3_8b_config = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
WARMUP  = 5
REPEATS = 20

def cuda_time_ms(fn, warmup=WARMUP, repeats=REPEATS):
    """Return median kernel wall-time in milliseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def prefill_flops(batch, H_qo, seq_len, head_dim, causal=True):
    """
    FLOPs for one attention layer of prefill.
      Q @ K^T  :  2 * B * H_qo * S * S * d
      scores@V :  2 * B * H_qo * S * S * d
      causal => ~1/2 of full attention
    """
    raw = 4 * batch * H_qo * seq_len * seq_len * head_dim
    return raw / 2 if causal else raw


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_torch_sdpa(batch, seq_len, H_qo, H_kv, head_dim):
    """
    torch.nn.functional.scaled_dot_product_attention
    q : (B, H_qo, S, d)
    k : (B, H_kv, S, d)  -- PyTorch handles GQA expansion internally (scaled_dot_product_attention with enable_gqa)

    Returns nan if the sequence length exceeds available VRAM (torch SDPA materializes
    the full S×S score matrix and OOMs at large seq_lens; FlashInfer does not).
    """
    try:
        q = torch.randn(batch, H_qo, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, H_kv, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, H_kv, seq_len, head_dim, device="cuda", dtype=torch.float16)

        def run():
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        t_ms = cuda_time_ms(run)
        flops = prefill_flops(batch, H_qo, seq_len, head_dim, causal=True)
        return flops / (t_ms * 1e-3) / 1e12
    except torch.OutOfMemoryError:
        print(f"    [torch SDPA OOM at seq_len={seq_len}, skipping]")
        return float("nan")
    finally:
        torch.cuda.empty_cache()


def bench_flashinfer_prefill(batch, seq_len, H_qo, H_kv, head_dim):
    """
    flashinfer.BatchPrefillWithRaggedKVCacheWrapper
    q  : (total_tokens, H_qo, d)
    kv : (total_tokens, H_kv, d)
    """
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")  # 256 MiB
    wrapper   = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    total = batch * seq_len
    q = torch.randn(total, H_qo, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(total, H_kv, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(total, H_kv, head_dim, device="cuda", dtype=torch.float16)

    # Uniform-length indptr  [0, S, 2S, ..., B*S]
    qo_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        H_qo,
        H_kv,
        head_dim,
        causal=True,
    )

    def run():
        wrapper.run(q, k, v)

    t_ms = cuda_time_ms(run)
    flops = prefill_flops(batch, H_qo, seq_len, head_dim, causal=True)
    tflops = flops / (t_ms * 1e-3) / 1e12
    return tflops


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------

def sweep_batch(config, batch_sizes, fixed_seq_len=512):
    H_qo     = config["num_attention_heads"]
    H_kv     = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // H_qo

    torch_tflops = []
    fi_tflops    = []
    for bs in batch_sizes:
        torch_tflops.append(bench_torch_sdpa        (bs, fixed_seq_len, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        fi_tflops   .append(bench_flashinfer_prefill(bs, fixed_seq_len, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        t_str = f"{torch_tflops[-1]:.2f}" if not np.isnan(torch_tflops[-1]) else "OOM"
        print(f"  bs={bs:4d}  torch={t_str} TF  fi={fi_tflops[-1]:.2f} TF")
    return torch_tflops, fi_tflops


def sweep_seq(config, seq_lens, fixed_batch=1):
    H_qo     = config["num_attention_heads"]
    H_kv     = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // H_qo

    torch_tflops = []
    fi_tflops    = []
    for sl in seq_lens:
        torch_tflops.append(bench_torch_sdpa        (fixed_batch, sl, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        fi_tflops   .append(bench_flashinfer_prefill(fixed_batch, sl, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        t_str = f"{torch_tflops[-1]:.2f}" if not np.isnan(torch_tflops[-1]) else "OOM"
        print(f"  seq={sl:5d}  torch={t_str} TF  fi={fi_tflops[-1]:.2f} TF")
    return torch_tflops, fi_tflops


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_seq(p_llama3, llama3_1b_sdpa, llama3_1b_flashinfer, llama3_3b_sdpa, llama3_3b_flashinfer, llama3_8b_sdpa, llama3_8b_flashinfer):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    # LLaMA3-1B plot
    mask = ~np.isnan(llama3_1b_sdpa)
    axs[0].plot(p_llama3[mask], llama3_1b_sdpa[mask], label='PyTorch SDPA', marker='o')
    axs[0].plot(p_llama3, llama3_1b_flashinfer, label='FlashInfer', marker='x')
    axs[0].set_xscale('log', base=2)
    axs[0].set_title(models[0])
    axs[0].set_xlabel('p (sequence length)')
    axs[0].set_ylabel('Compute Utilization (TFLOPs/s)')
    axs[0].set_xticks(p_llama3)
    axs[0].set_xticklabels([str(p) for p in p_llama3])
    axs[0].legend()
    axs[0].grid(True, which='both')

    # LLaMA3-3B plot
    mask = ~np.isnan(llama3_3b_sdpa)
    axs[1].plot(p_llama3[mask], llama3_3b_sdpa[mask], label='PyTorch SDPA', marker='o')
    axs[1].plot(p_llama3, llama3_3b_flashinfer, label='FlashInfer', marker='x')
    axs[1].set_xscale('log', base=2)
    axs[1].set_title(models[1])
    axs[1].set_xlabel('p (sequence length)')
    axs[1].set_xticks(p_llama3)
    axs[1].set_xticklabels([str(p) for p in p_llama3])
    axs[1].legend()
    axs[1].grid(True, which='both')

    # LLaMA3-8B plot
    mask = ~np.isnan(llama3_8b_sdpa)
    axs[2].plot(p_llama3[mask], llama3_8b_sdpa[mask], label='PyTorch SDPA', marker='o')
    axs[2].plot(p_llama3, llama3_8b_flashinfer, label='FlashInfer', marker='x')
    axs[2].set_xscale('log', base=2)
    axs[2].set_title(models[2])
    axs[2].set_xlabel('p (sequence length)')
    axs[2].set_xticks(p_llama3)
    axs[2].set_xticklabels([str(p) for p in p_llama3])
    axs[2].legend()
    axs[2].grid(True, which='both')

    # Overall figure title and layout
    fig.suptitle('Prefill Attention Compute Utilization', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('prefill_attention_by_seq_len.png', dpi=300)
    plt.close()
    print("Saved → prefill_attention_by_seq_len.png")

def plot_batch(batch_sizes, llama3_1b_sdpa, llama3_1b_flashinfer, llama3_3b_sdpa, llama3_3b_flashinfer, llama3_8b_sdpa, llama3_8b_flashinfer, fixed_seq=1024):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']
    b_arr = np.array(batch_sizes)

    for ax, sdpa, fi, name in zip(
        axs,
        [llama3_1b_sdpa, llama3_3b_sdpa, llama3_8b_sdpa],
        [llama3_1b_flashinfer, llama3_3b_flashinfer, llama3_8b_flashinfer],
        models,
    ):
        mask = ~np.isnan(np.array(sdpa))
        ax.plot(b_arr[mask], np.array(sdpa)[mask], label='PyTorch SDPA', marker='o')
        ax.plot(b_arr, np.array(fi), label='FlashInfer', marker='x')
        ax.set_xscale('log', base=2)
        ax.set_title(name)
        ax.set_xlabel('Batch Size')
        ax.set_xticks(b_arr)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.legend()
        ax.grid(True, which='both')

    axs[0].set_ylabel('Compute Utilization (TFLOPs/s)')
    fig.suptitle(f'Prefill Attention Compute Utilization (p={fixed_seq})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('prefill_attention_by_batch_size.png', dpi=300)
    plt.close()
    print('Saved → prefill_attention_by_batch_size.png')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- sequence lengths (shared x-axis for all models) ----
    p_llama3 = [2**i for i in range(7, 16)]   # 128 … 32768
    fixed_batch = 1

    # ---- run sweeps ----
    print("\n[Seq-len sweep] LLaMA3-1B")
    llama3_1b_sdpa,       llama3_1b_flashinfer       = sweep_seq(llama3_1b_config, p_llama3, fixed_batch)
    torch.cuda.empty_cache()

    print("\n[Seq-len sweep] LLaMA3-3B")
    llama3_3b_sdpa,       llama3_3b_flashinfer       = sweep_seq(llama3_3b_config, p_llama3, fixed_batch)
    torch.cuda.empty_cache()

    print("\n[Seq-len sweep] LLaMA3-8B")
    llama3_8b_sdpa,       llama3_8b_flashinfer       = sweep_seq(llama3_8b_config, p_llama3, fixed_batch)
    torch.cuda.empty_cache()

    # Convert to numpy arrays; keep NaN for OOM points
    p_llama3             = np.array(p_llama3)
    llama3_1b_sdpa       = np.array(llama3_1b_sdpa)
    llama3_1b_flashinfer = np.array(llama3_1b_flashinfer)
    llama3_3b_sdpa       = np.array(llama3_3b_sdpa)
    llama3_3b_flashinfer = np.array(llama3_3b_flashinfer)
    llama3_8b_sdpa       = np.array(llama3_8b_sdpa)
    llama3_8b_flashinfer = np.array(llama3_8b_flashinfer)

    # ---- Plotting: seq-len sweep ----
    plot_seq(p_llama3, llama3_1b_sdpa, llama3_1b_flashinfer, llama3_3b_sdpa, llama3_3b_flashinfer, llama3_8b_sdpa, llama3_8b_flashinfer)

    # ---- Batch-size sweep, p=1024 ----
    batch_sizes = [2**i for i in range(6)]   # 1, 2, 4, 8, 16, 32
    fixed_seq   = 1024

    print("\n[Batch sweep] LLaMA3-1B")
    b_1b_sdpa, b_1b_fi = sweep_batch(llama3_1b_config, batch_sizes, fixed_seq_len=fixed_seq)
    torch.cuda.empty_cache()

    print("\n[Batch sweep] LLaMA3-3B")
    b_3b_sdpa, b_3b_fi = sweep_batch(llama3_3b_config, batch_sizes, fixed_seq_len=fixed_seq)
    torch.cuda.empty_cache()

    print("\n[Batch sweep] LLaMA3-8B")
    b_8b_sdpa, b_8b_fi = sweep_batch(llama3_8b_config, batch_sizes, fixed_seq_len=fixed_seq)
    torch.cuda.empty_cache()

    # ---- Plotting: batch sweep ----
    plot_batch(batch_sizes, b_1b_sdpa, b_1b_fi, b_3b_sdpa, b_3b_fi, b_8b_sdpa, b_8b_fi, fixed_seq=fixed_seq)
