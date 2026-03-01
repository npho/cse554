import flashinfer
import numpy as np
import matplotlib.pyplot as plt
import torch

# Reference config for each model
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


def create_decode_inputs(batch_size, num_qo_heads, num_kv_heads, context_len, head_dim):
    # Single query token per batch entry, full KV cache of context_len tokens
    Q = torch.randn(batch_size, num_qo_heads, head_dim,
                    dtype=torch.float16, device='cuda')
    K = torch.randn(batch_size, context_len, num_kv_heads, head_dim,
                    dtype=torch.float16, device='cuda')
    V = torch.randn(batch_size, context_len, num_kv_heads, head_dim,
                    dtype=torch.float16, device='cuda')
    return Q, K, V


def benchmark_pytorch_sdpa_decode(Q, K, V, warmup=10, iters=100):
    # SDPA expects (batch, heads, seq, head_dim)
    # Q: (batch, num_qo_heads, head_dim) -> (batch, num_qo_heads, 1, head_dim)
    # K: (batch, context_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, context_len, head_dim)
    Q_sdpa = Q.unsqueeze(2)
    K_sdpa = K.permute(0, 2, 1, 3).contiguous()
    V_sdpa = V.permute(0, 2, 1, 3).contiguous()

    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa,
                                                             enable_gqa=True)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        output = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa,
                                                                  enable_gqa=True)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


def benchmark_flashinfer_decode(Q, K, V, warmup=10, iters=100):
    # single_decode_with_kv_cache expects per-sequence tensors:
    # q: (num_qo_heads, head_dim)
    # k: (context_len, num_kv_heads, head_dim)
    # v: (context_len, num_kv_heads, head_dim)
    # Loop over the batch dimension to simulate batched decode.
    batch_size = Q.shape[0]

    def run_batch():
        for i in range(batch_size):
            flashinfer.decode.single_decode_with_kv_cache(Q[i], K[i], V[i])

    # Warmup
    for _ in range(warmup):
        run_batch()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        run_batch()
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms


def compute_memory_bandwidth_GBs_decode(batch_size, context_len, num_qo_heads, num_kv_heads,
                                        head_dim, time_ms):
    """Return achieved memory bandwidth in GB/s for a decode attention step.

    Bytes accessed (float16 = 2 bytes per element):
      - Read  Q:      batch * num_qo_heads * head_dim
      - Read  K:      batch * context_len  * num_kv_heads * head_dim
      - Read  V:      batch * context_len  * num_kv_heads * head_dim
      - Write output: batch * num_qo_heads * head_dim
    """
    bytes_per_elem = 2  # float16
    q_bytes = batch_size * num_qo_heads * head_dim * bytes_per_elem
    k_bytes = batch_size * context_len * num_kv_heads * head_dim * bytes_per_elem
    v_bytes = batch_size * context_len * num_kv_heads * head_dim * bytes_per_elem
    out_bytes = batch_size * num_qo_heads * head_dim * bytes_per_elem
    total_bytes = q_bytes + k_bytes + v_bytes + out_bytes

    return total_bytes / (time_ms * 1e-3) / 1e9


def benchmark_context_lens(config, context_lens: list[int], batch_size: int):
    bw_sdpa_by_len: list[float] = []
    bw_flashinfer_by_len: list[float] = []
    for context_len in context_lens:
        print(f"Benchmarking (context_len={context_len}, batch_size={batch_size})...")
        num_qo_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_qo_heads

        Q, K, V = create_decode_inputs(batch_size, num_qo_heads, num_kv_heads, context_len, head_dim)

        time_sdpa = benchmark_pytorch_sdpa_decode(Q, K, V)
        time_flashinfer = benchmark_flashinfer_decode(Q, K, V)

        bw_sdpa = compute_memory_bandwidth_GBs_decode(
            batch_size, context_len, num_qo_heads, num_kv_heads, head_dim, time_sdpa)
        bw_flashinfer = compute_memory_bandwidth_GBs_decode(
            batch_size, context_len, num_qo_heads, num_kv_heads, head_dim, time_flashinfer)

        print(f"  PyTorch SDPA:  {time_sdpa:.3f} ms | mem bw {bw_sdpa:.2f} GB/s")
        print(f"  FlashInfer:    {time_flashinfer:.3f} ms | mem bw {bw_flashinfer:.2f} GB/s")

        bw_sdpa_by_len.append(bw_sdpa)
        bw_flashinfer_by_len.append(bw_flashinfer)

    return bw_sdpa_by_len, bw_flashinfer_by_len


def benchmark_batch_sizes_decode(config, batch_sizes: list[int], context_len: int):
    bw_sdpa_by_batch: list[float] = []
    bw_flashinfer_by_batch: list[float] = []
    for batch_size in batch_sizes:
        print(f"Benchmarking (context_len={context_len}, batch_size={batch_size})...")
        num_qo_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_qo_heads

        Q, K, V = create_decode_inputs(batch_size, num_qo_heads, num_kv_heads, context_len, head_dim)

        time_sdpa = benchmark_pytorch_sdpa_decode(Q, K, V)
        time_flashinfer = benchmark_flashinfer_decode(Q, K, V)

        bw_sdpa = compute_memory_bandwidth_GBs_decode(
            batch_size, context_len, num_qo_heads, num_kv_heads, head_dim, time_sdpa)
        bw_flashinfer = compute_memory_bandwidth_GBs_decode(
            batch_size, context_len, num_qo_heads, num_kv_heads, head_dim, time_flashinfer)

        print(f"  PyTorch SDPA:  {time_sdpa:.3f} ms | mem bw {bw_sdpa:.2f} GB/s")
        print(f"  FlashInfer:    {time_flashinfer:.3f} ms | mem bw {bw_flashinfer:.2f} GB/s")

        bw_sdpa_by_batch.append(bw_sdpa)
        bw_flashinfer_by_batch.append(bw_flashinfer)

    return bw_sdpa_by_batch, bw_flashinfer_by_batch


def _add_subplot(ax, xs, sdpa_vals, fi_vals, xlabel, ylabel, title, xscale_base=2):
    ax.plot(xs, sdpa_vals, label='PyTorch SDPA', marker='o')
    ax.plot(xs, fi_vals, label='FlashInfer', marker='x')
    ax.set_xscale('log', base=xscale_base)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.legend()
    ax.grid(True, which='both')


def benchmark_models_by_context_len():
    # Context lengths (powers of 2)
    c_llama3 = 2 ** np.arange(7, 16)   # 2^7 to 2^15

    # Batch size
    batch_size = 1

    # Benchmark LLAMA3-1B
    llama3_1b_bw_sdpa, llama3_1b_bw_fi = \
        benchmark_context_lens(llama3_1b_config, c_llama3, batch_size)
    # Benchmark LLAMA3-3B
    llama3_3b_bw_sdpa, llama3_3b_bw_fi = \
        benchmark_context_lens(llama3_3b_config, c_llama3, batch_size)
    # Benchmark LLAMA3-8B
    llama3_8b_bw_sdpa, llama3_8b_bw_fi = \
        benchmark_context_lens(llama3_8b_config, c_llama3, batch_size)

    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    # --- Memory bandwidth plot ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, sdpa_bw, fi_bw, model in zip(
            axs,
            [llama3_1b_bw_sdpa, llama3_3b_bw_sdpa, llama3_8b_bw_sdpa],
            [llama3_1b_bw_fi, llama3_3b_bw_fi, llama3_8b_bw_fi],
            models):
        _add_subplot(ax, c_llama3, sdpa_bw, fi_bw,
                     xlabel='c (context length)',
                     ylabel='Memory Bandwidth (GB/s)',
                     title=model)
    fig.suptitle('Decode Attention Memory Bandwidth', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_by_context_len.png', dpi=300)
    plt.close(fig)


def benchmark_models_by_batch_size():
    # Context length
    context_len = 1024

    # Batch sizes (powers of 2)
    batch_sizes = 2 ** np.arange(0, 6)   # 1 to 32

    # Benchmark LLAMA3-1B
    llama3_1b_bw_sdpa, llama3_1b_bw_fi = \
        benchmark_batch_sizes_decode(llama3_1b_config, batch_sizes, context_len)
    # Benchmark LLAMA3-3B
    llama3_3b_bw_sdpa, llama3_3b_bw_fi = \
        benchmark_batch_sizes_decode(llama3_3b_config, batch_sizes, context_len)
    # Benchmark LLAMA3-8B
    llama3_8b_bw_sdpa, llama3_8b_bw_fi = \
        benchmark_batch_sizes_decode(llama3_8b_config, batch_sizes, context_len)

    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    # --- Memory bandwidth plot ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, sdpa_bw, fi_bw, model in zip(
            axs,
            [llama3_1b_bw_sdpa, llama3_3b_bw_sdpa, llama3_8b_bw_sdpa],
            [llama3_1b_bw_fi, llama3_3b_bw_fi, llama3_8b_bw_fi],
            models):
        _add_subplot(ax, batch_sizes, sdpa_bw, fi_bw,
                     xlabel='Batch Size',
                     ylabel='Memory Bandwidth (GB/s)',
                     title=model)
    fig.suptitle('Decode Attention Memory Bandwidth by Batch Size', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_by_batch_size.png', dpi=300)
    plt.close(fig)


def benchmark_flashinfer_paged_decode(batch_size, num_qo_heads, num_kv_heads, context_len,
                                      head_dim, page_size, warmup=10, iters=100):
    """Benchmark FlashInfer BatchDecodeWithPagedKVCacheWrapper for a given page size.

    KV cache layout (NHD): [total_pages, 2, page_size, num_kv_heads, head_dim]
    Q layout: [batch_size, num_qo_heads, head_dim]
    """
    assert context_len % page_size == 0, "context_len must be divisible by page_size"
    pages_per_seq = context_len // page_size
    total_pages = batch_size * pages_per_seq

    # Build paged KV tensors
    kv_cache = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.float16, device='cuda')
    q = torch.randn(batch_size, num_qo_heads, head_dim,
                    dtype=torch.float16, device='cuda')

    # Build page table structures
    indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device='cuda') * pages_per_seq
    indices = torch.arange(total_pages, dtype=torch.int32, device='cuda')
    last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device='cuda')

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(indptr, indices, last_page_len,
                 num_qo_heads, num_kv_heads, head_dim, page_size,
                 data_type=torch.float16)

    # Warmup
    for _ in range(warmup):
        wrapper.run(q, kv_cache)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        wrapper.run(q, kv_cache)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_page_sizes(config, page_sizes: list[int], batch_size: int, context_len: int):
    bw_by_page_size: list[float] = []
    for page_size in page_sizes:
        print(f"Benchmarking (page_size={page_size}, batch_size={batch_size}, "
              f"context_len={context_len})...")
        num_qo_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_qo_heads

        time_ms = benchmark_flashinfer_paged_decode(
            batch_size, num_qo_heads, num_kv_heads, context_len, head_dim, page_size)
        bw = compute_memory_bandwidth_GBs_decode(
            batch_size, context_len, num_qo_heads, num_kv_heads, head_dim, time_ms)

        print(f"  FlashInfer paged (page_size={page_size}): "
              f"{time_ms:.3f} ms | mem bw {bw:.2f} GB/s")
        bw_by_page_size.append(bw)
    return bw_by_page_size


def benchmark_models_by_page_size():
    batch_size = 128
    context_len = 1024
    page_sizes = [1, 2, 4, 8, 16]

    llama3_1b_bw = benchmark_page_sizes(llama3_1b_config, page_sizes, batch_size, context_len)
    llama3_3b_bw = benchmark_page_sizes(llama3_3b_config, page_sizes, batch_size, context_len)
    llama3_8b_bw = benchmark_page_sizes(llama3_8b_config, page_sizes, batch_size, context_len)

    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, bw, model in zip(axs, [llama3_1b_bw, llama3_3b_bw, llama3_8b_bw], models):
        ax.plot(page_sizes, bw, marker='o', label='FlashInfer Paged')
        ax.set_xscale('log', base=2)
        ax.set_title(model)
        ax.set_xlabel('Page Size')
        ax.set_ylabel('Memory Bandwidth (GB/s)')
        ax.set_xticks(page_sizes)
        ax.set_xticklabels([str(p) for p in page_sizes])
        ax.legend()
        ax.grid(True, which='both')

    fig.suptitle(
        f'Decode Attention Memory Bandwidth by Page Size '
        f'(batch={batch_size}, context={context_len})',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_by_page_size.png', dpi=300)
    plt.close(fig)


benchmark_models_by_context_len()
benchmark_models_by_batch_size()
benchmark_models_by_page_size()
