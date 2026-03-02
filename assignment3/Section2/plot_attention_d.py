import math

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


def decode_bytes(batch, H_qo, H_kv, context_len, head_dim):
    """
    HBM bytes read during one decode attention step (FP16 = 2 bytes/element).
      Q  : batch * H_qo * 1 * head_dim
      K  : batch * H_kv * context_len * head_dim
      V  : batch * H_kv * context_len * head_dim
    """
    return 2 * (batch * H_qo * head_dim
                + 2 * batch * H_kv * context_len * head_dim)


# ---------------------------------------------------------------------------
# Benchmark: torch SDPA decode
# ---------------------------------------------------------------------------

def bench_torch_sdpa_decode(batch, context_len, H_qo, H_kv, head_dim):
    """
    q : (B, H_qo, 1, d)
    k : (B, H_kv, c, d)
    v : (B, H_kv, c, d)
    Returns GB/s, or nan on OOM.
    """
    try:
        q = torch.randn(batch, H_qo, 1,           head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, H_kv, context_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, H_kv, context_len, head_dim, device="cuda", dtype=torch.float16)

        def run():
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)

        t_ms = cuda_time_ms(run)
        bw   = decode_bytes(batch, H_qo, H_kv, context_len, head_dim)
        return bw / (t_ms * 1e-3) / 1e9
    except torch.OutOfMemoryError:
        print(f"    [torch SDPA OOM at batch={batch}, c={context_len}]")
        return float("nan")
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Benchmark: FlashInfer paged-KV decode
# ---------------------------------------------------------------------------

def bench_flashinfer_decode(batch, context_len, H_qo, H_kv, head_dim, page_size=16):
    """
    Uses BatchDecodeWithPagedKVCacheWrapper with NHD layout.
    kv_cache : (total_pages, 2, page_size, H_kv, head_dim)
    q        : (batch, H_qo, head_dim)
    Returns GB/s.
    """
    workspace     = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    wrapper       = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace, "NHD", use_tensor_cores=True)

    pages_per_req     = math.ceil(context_len / page_size)
    last_page_len_val = (context_len - 1) % page_size + 1
    total_pages       = batch * pages_per_req

    kv_cache = torch.randn(total_pages, 2, page_size, H_kv, head_dim,
                           device="cuda", dtype=torch.float16)
    q        = torch.randn(batch, H_qo, head_dim, device="cuda", dtype=torch.float16)

    indptr        = torch.arange(0, (batch + 1) * pages_per_req, pages_per_req,
                                 dtype=torch.int32, device="cuda")
    indices       = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full((batch,), last_page_len_val,
                               dtype=torch.int32, device="cuda")

    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        H_qo,
        H_kv,
        head_dim,
        page_size,
        data_type=torch.float16,
    )

    def run():
        wrapper.run(q, kv_cache)

    t_ms = cuda_time_ms(run)
    bw   = decode_bytes(batch, H_qo, H_kv, context_len, head_dim)
    return bw / (t_ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_context(config, context_lens, fixed_batch=1, page_size=16):
    H_qo     = config["num_attention_heads"]
    H_kv     = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // H_qo

    sdpa_bw = []
    fi_bw   = []
    for c in context_lens:
        sdpa_bw.append(bench_torch_sdpa_decode(fixed_batch, c, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        fi_bw.append(bench_flashinfer_decode(fixed_batch, c, H_qo, H_kv, head_dim, page_size))
        torch.cuda.empty_cache()
        s = f"{sdpa_bw[-1]:.1f}" if not np.isnan(sdpa_bw[-1]) else "OOM"
        print(f"  c={c:6d}  torch={s} GB/s  fi={fi_bw[-1]:.1f} GB/s")
    return sdpa_bw, fi_bw


def sweep_batch(config, batch_sizes, fixed_context=1024, page_size=16):
    H_qo     = config["num_attention_heads"]
    H_kv     = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // H_qo

    sdpa_bw = []
    fi_bw   = []
    for bs in batch_sizes:
        sdpa_bw.append(bench_torch_sdpa_decode(bs, fixed_context, H_qo, H_kv, head_dim))
        torch.cuda.empty_cache()
        fi_bw.append(bench_flashinfer_decode(bs, fixed_context, H_qo, H_kv, head_dim, page_size))
        torch.cuda.empty_cache()
        s = f"{sdpa_bw[-1]:.1f}" if not np.isnan(sdpa_bw[-1]) else "OOM"
        print(f"  bs={bs:4d}  torch={s} GB/s  fi={fi_bw[-1]:.1f} GB/s")
    return sdpa_bw, fi_bw


def sweep_page_size(config, page_sizes, fixed_batch=128, fixed_context=1024):
    H_qo     = config["num_attention_heads"]
    H_kv     = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // H_qo

    fi_bw = []
    for ps in page_sizes:
        fi_bw.append(bench_flashinfer_decode(fixed_batch, fixed_context, H_qo, H_kv, head_dim, ps))
        torch.cuda.empty_cache()
        print(f"  page_size={ps:3d}  fi={fi_bw[-1]:.1f} GB/s")
    return fi_bw


# ---------------------------------------------------------------------------
# Shared 1×3 subplot helper
# ---------------------------------------------------------------------------

def plot_1x3(x_vals, data_per_model, x_label, y_label, title, save_path,
             x_scale="log", x_base=2, sdpa_key="sdpa", fi_key="fi",
             fi_only=False):
    """
    data_per_model: list of 3 dicts, one per model.
      Each dict must have 'sdpa' and 'fi' lists (or just 'fi' when fi_only=True).
    """
    model_names = ["LLaMA3-1B", "LLaMA3-3B", "LLaMA3-8B"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    x_arr = np.array(x_vals)

    for i, (ax, name, data) in enumerate(zip(axs, model_names, data_per_model)):
        if not fi_only:
            sdpa_arr = np.array(data[sdpa_key])
            mask     = ~np.isnan(sdpa_arr)
            ax.plot(x_arr[mask], sdpa_arr[mask], label="PyTorch SDPA", marker="o")
        fi_arr = np.array(data[fi_key])
        ax.plot(x_arr, fi_arr, label="FlashInfer", marker="x")

        if x_scale == "log":
            ax.set_xscale("log", base=x_base)
        ax.set_title(name)
        ax.set_xlabel(x_label)
        if i == 0:
            ax.set_ylabel(y_label)
        ax.set_xticks(x_arr)
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.legend()
        ax.grid(True, which="both")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs = {
        "LLaMA3-1B": llama3_1b_config,
        "LLaMA3-3B": llama3_3b_config,
        "LLaMA3-8B": llama3_8b_config,
    }

    # -----------------------------------------------------------------------
    # Plot 1: vary context length, batch=1
    # -----------------------------------------------------------------------
    context_lens = [2**i for i in range(7, 16)]   # 128 … 32768
    fixed_batch  = 1

    ctx_data = []
    for name, cfg in configs.items():
        print(f"\n[Context sweep] {name}")
        sdpa, fi = sweep_context(cfg, context_lens, fixed_batch=fixed_batch)
        ctx_data.append({"sdpa": sdpa, "fi": fi})
    torch.cuda.empty_cache()

    plot_1x3(
        x_vals         = context_lens,
        data_per_model = ctx_data,
        x_label        = "c (context length)",
        y_label        = "Memory Bandwidth (GB/s)",
        title          = f"Decode Attention — GB/s vs Context Length  (batch={fixed_batch})",
        save_path      = "decode_attention_by_context_len.png",
        x_scale        = "log",
        x_base         = 2,
    )

    # -----------------------------------------------------------------------
    # Plot 2: vary batch size, c=1024
    # -----------------------------------------------------------------------
    batch_sizes   = [2**i for i in range(7)]   # 1, 2, …, 64
    fixed_context = 1024

    batch_data = []
    for name, cfg in configs.items():
        print(f"\n[Batch sweep] {name}")
        sdpa, fi = sweep_batch(cfg, batch_sizes, fixed_context=fixed_context)
        batch_data.append({"sdpa": sdpa, "fi": fi})
    torch.cuda.empty_cache()

    plot_1x3(
        x_vals         = batch_sizes,
        data_per_model = batch_data,
        x_label        = "Batch Size",
        y_label        = "Memory Bandwidth (GB/s)",
        title          = f"Decode Attention — GB/s vs Batch Size  (c={fixed_context})",
        save_path      = "decode_attention_by_batch_size.png",
        x_scale        = "log",
        x_base         = 2,
    )

    # -----------------------------------------------------------------------
    # Plot 3: vary page size, batch=128, c=1024, FlashInfer only
    # -----------------------------------------------------------------------
    page_sizes   = [1, 2, 4, 8, 16]
    fixed_batch  = 128
    fixed_context = 1024

    ps_data = []
    for name, cfg in configs.items():
        print(f"\n[Page-size sweep] {name}")
        fi = sweep_page_size(cfg, page_sizes, fixed_batch=fixed_batch, fixed_context=fixed_context)
        ps_data.append({"fi": fi})
    torch.cuda.empty_cache()

    plot_1x3(
        x_vals         = page_sizes,
        data_per_model = ps_data,
        x_label        = "Page Size",
        y_label        = "Memory Bandwidth (GB/s)",
        title          = f"Decode Attention — GB/s vs Page Size  (FlashInfer, batch={fixed_batch}, c={fixed_context})",
        save_path      = "decode_attention_by_page_size.png",
        x_scale        = "linear",
        fi_only        = True,
    )
