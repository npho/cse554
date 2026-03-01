import flashinfer
import numpy as np
import matplotlib.pyplot as plt
import torch

#Reference config for each model
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


def create_attention_inputs(batch_size, num_qo_heads, num_kv_heads, seq_len, head_dim):
    Q = torch.randn(batch_size, num_qo_heads, seq_len, head_dim,
                    dtype=torch.float16, device='cuda')
    K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                    dtype=torch.float16, device='cuda')
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                    dtype=torch.float16, device='cuda')
    return Q, K, V

def benchmark_pytorch_sdpa(Q, K, V, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms

def benchmark_flashinfer(Q, K, V, warmup=10, iters=100):
    # FlashInfer expects different tensor layout
    # Reshape to (batch_size, seq_len, num_heads, head_dim)
    Q = Q.transpose(1, 2).contiguous()
    K = K.transpose(1, 2).contiguous()
    V = V.transpose(1, 2).contiguous()
    
    # Warmup
    for _ in range(warmup):
        _ = flashinfer.single_prefill_with_kv_cache(Q, K, V)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        output = flashinfer.single_prefill_with_kv_cache(Q, K, V)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    return elapsed_ms

def compute_tflops(batch_size, seq_len, num_heads, head_dim, time_ms):
    hidden_size = num_heads * head_dim
    flops = batch_size * 4 * seq_len * seq_len * hidden_size
    tflops = (flops / (time_ms * 1e-3)) / 1e12
    return tflops

def benchmark_seq_lens(config, seq_lens: list[int], batch_size: int):
    tflops_sdpa_by_len: list[float] = []
    tflops_flashinfer_by_len: list[float] = []
    for seq_len in seq_lens:
        print(f"Benchmarking (seq_len={seq_len}, batch_size={batch_size})...")
        num_qo_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_qo_heads

        try:
            Q, K, V = create_attention_inputs(batch_size, num_qo_heads, num_kv_heads, seq_len, head_dim)
        except torch.OutOfMemoryError:
            print(f"  OOM creating tensors at seq_len={seq_len}, skipping...")
            torch.cuda.empty_cache()
            tflops_sdpa_by_len.append(float('nan'))
            tflops_flashinfer_by_len.append(float('nan'))
            continue

        try:
            time_sdpa = benchmark_pytorch_sdpa(Q, K, V)
            tflops_sdpa = compute_tflops(batch_size, seq_len, num_qo_heads, head_dim, time_sdpa)
            print(f"PyTorch SDPA: {time_sdpa:.2f} ms ({tflops_sdpa:.2f} TFLOPs)")
        except torch.OutOfMemoryError:
            print(f"  SDPA OOM at seq_len={seq_len}, recording NaN")
            torch.cuda.empty_cache()
            tflops_sdpa = float('nan')

        try:
            time_flashinfer = benchmark_flashinfer(Q, K, V)
            tflops_flashinfer = compute_tflops(batch_size, seq_len, num_qo_heads, head_dim, time_flashinfer)
            print(f"FlashInfer: {time_flashinfer:.2f} ms ({tflops_flashinfer:.2f} TFLOPs)")
        except torch.OutOfMemoryError:
            print(f"  FlashInfer OOM at seq_len={seq_len}, recording NaN")
            torch.cuda.empty_cache()
            tflops_flashinfer = float('nan')

        tflops_sdpa_by_len.append(tflops_sdpa)
        tflops_flashinfer_by_len.append(tflops_flashinfer)

        del Q, K, V
        torch.cuda.empty_cache()

    return tflops_sdpa_by_len, tflops_flashinfer_by_len

def benchmark_models_by_seq_len():
    # Sequence lengths (powers of 2)
    p_llama3 = 2 ** np.arange(7, 16)   # 2^7 to 2^15

    # Batch size
    batch_size = 1

    # Benchmark LLAMA3-1B
    llama3_1b_sdpa, llama3_1b_flashinfer = benchmark_seq_lens(llama3_1b_config, p_llama3, batch_size)
    # Benchmark LLAMA3-3B
    llama3_3b_sdpa, llama3_3b_flashinfer = benchmark_seq_lens(llama3_3b_config, p_llama3, batch_size)
    # Benchmark LLAMA3-8B
    llama3_8b_sdpa, llama3_8b_flashinfer = benchmark_seq_lens(llama3_8b_config, p_llama3, batch_size)

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    # LLaMA3-1B plot
    axs[0].plot(p_llama3, llama3_1b_sdpa, label='PyTorch SDPA', marker='o')
    axs[0].plot(p_llama3, llama3_1b_flashinfer, label='FlashInfer', marker='x')
    axs[0].set_xscale('log', base=2)
    axs[0].set_title(models[0])
    axs[0].set_xlabel('p (sequence length)')
    axs[0].set_ylabel('Compute Utilization (TFLOPs)')
    axs[0].set_xticks(p_llama3)
    axs[0].set_xticklabels([str(p) for p in p_llama3])
    axs[0].legend()
    axs[0].grid(True, which='both')

    # LLaMA3-3B plot
    axs[1].plot(p_llama3, llama3_3b_sdpa, label='PyTorch SDPA', marker='o')
    axs[1].plot(p_llama3, llama3_3b_flashinfer, label='FlashInfer', marker='x')
    axs[1].set_xscale('log', base=2)
    axs[1].set_title(models[1])
    axs[1].set_xlabel('p (sequence length)')
    axs[1].set_xticks(p_llama3)
    axs[1].set_xticklabels([str(p) for p in p_llama3])
    axs[1].legend()
    axs[1].grid(True, which='both')

    # LLaMA3-8B plot
    axs[2].plot(p_llama3, llama3_8b_sdpa, label='PyTorch SDPA', marker='o')
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


def benchmark_batch_sizes(config, batch_sizes: list[int], seq_len: int):
    tflops_sdpa_by_batch: list[float] = []
    tflops_flashinfer_by_batch: list[float] = []
    for batch_size in batch_sizes:
        print(f"Benchmarking (seq_len={seq_len}, batch_size={batch_size})...")
        num_qo_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_qo_heads

        try:
            Q, K, V = create_attention_inputs(batch_size, num_qo_heads, num_kv_heads, seq_len, head_dim)
        except torch.OutOfMemoryError:
            print(f"  OOM creating tensors at batch_size={batch_size}, skipping...")
            torch.cuda.empty_cache()
            tflops_sdpa_by_batch.append(float('nan'))
            tflops_flashinfer_by_batch.append(float('nan'))
            continue

        try:
            time_sdpa = benchmark_pytorch_sdpa(Q, K, V)
            tflops_sdpa = compute_tflops(batch_size, seq_len, num_qo_heads, head_dim, time_sdpa)
            print(f"PyTorch SDPA: {time_sdpa:.2f} ms ({tflops_sdpa:.2f} TFLOPs)")
        except torch.OutOfMemoryError:
            print(f"  SDPA OOM at batch_size={batch_size}, recording NaN")
            torch.cuda.empty_cache()
            tflops_sdpa = float('nan')

        try:
            time_flashinfer = benchmark_flashinfer(Q, K, V)
            tflops_flashinfer = compute_tflops(batch_size, seq_len, num_qo_heads, head_dim, time_flashinfer)
            print(f"FlashInfer: {time_flashinfer:.2f} ms ({tflops_flashinfer:.2f} TFLOPs)")
        except torch.OutOfMemoryError:
            print(f"  FlashInfer OOM at batch_size={batch_size}, recording NaN")
            torch.cuda.empty_cache()
            tflops_flashinfer = float('nan')

        tflops_sdpa_by_batch.append(tflops_sdpa)
        tflops_flashinfer_by_batch.append(tflops_flashinfer)

        del Q, K, V
        torch.cuda.empty_cache()

    return tflops_sdpa_by_batch, tflops_flashinfer_by_batch

def benchmark_models_by_batch_size():
    # Sequence length
    seq_len = 1024

    # Batch sizes (powers of 2)
    batch_sizes = 2 ** np.arange(0, 6)   # 1 to 32

    # Benchmark LLAMA3-1B
    llama3_1b_sdpa, llama3_1b_flashinfer = benchmark_batch_sizes(llama3_1b_config, batch_sizes, seq_len)
    # Benchmark LLAMA3-3B
    llama3_3b_sdpa, llama3_3b_flashinfer = benchmark_batch_sizes(llama3_3b_config, batch_sizes, seq_len)
    # Benchmark LLAMA3-8B
    llama3_8b_sdpa, llama3_8b_flashinfer = benchmark_batch_sizes(llama3_8b_config, batch_sizes, seq_len)

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

    # LLaMA3-1B plot
    axs[0].plot(batch_sizes, llama3_1b_sdpa, label='PyTorch SDPA', marker='o')
    axs[0].plot(batch_sizes, llama3_1b_flashinfer, label='FlashInfer', marker='x')
    axs[0].set_xscale('log', base=2)
    axs[0].set_title(models[0])
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Compute Utilization (TFLOPs)')
    axs[0].set_xticks(batch_sizes)
    axs[0].set_xticklabels([str(b) for b in batch_sizes])
    axs[0].legend()
    axs[0].grid(True, which='both')

    # LLaMA3-3B plot
    axs[1].plot(batch_sizes, llama3_3b_sdpa, label='PyTorch SDPA', marker='o')
    axs[1].plot(batch_sizes, llama3_3b_flashinfer, label='FlashInfer', marker='x')
    axs[1].set_xscale('log', base=2)
    axs[1].set_title(models[1])
    axs[1].set_xlabel('Batch Size')
    axs[1].set_xticks(batch_sizes)
    axs[1].set_xticklabels([str(b) for b in batch_sizes])
    axs[1].legend()
    axs[1].grid(True, which='both')

    # LLaMA3-8B plot
    axs[2].plot(batch_sizes, llama3_8b_sdpa, label='PyTorch SDPA', marker='o')
    axs[2].plot(batch_sizes, llama3_8b_flashinfer, label='FlashInfer', marker='x')
    axs[2].set_xscale('log', base=2)
    axs[2].set_title(models[2])
    axs[2].set_xlabel('Batch Size')
    axs[2].set_xticks(batch_sizes)
    axs[2].set_xticklabels([str(b) for b in batch_sizes])
    axs[2].legend()
    axs[2].grid(True, which='both')

    # Overall figure title and layout
    fig.suptitle('Prefill Attention Compute Utilization by Batch Size', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('prefill_attention_by_batch_size.png', dpi=300)


benchmark_models_by_seq_len()
benchmark_models_by_batch_size()
