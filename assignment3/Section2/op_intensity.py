"""
Operational intensity formulas for attention mechanisms.

Operational intensity = FLOPs / bytes_read

We assume FP16/BF16 weights (2 bytes per element).
"""

import numpy as np

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


def prefill_op_intensity(H_qo: int, H_kv: int, d: int, p: int) -> float:
    """
    Operational intensity of prefill (encoding) attention.

    For each query head, attention requires:
      - Q @ K^T : 2 * p^2 * d FLOPs
      - Scores @ V : 2 * p^2 * d FLOPs
    Total FLOPs: 4 * H_qo * p^2 * d

    Memory reads (FP16 => 2 bytes/element):
      - Q : H_qo * p * d elements
      - K : H_kv * p * d elements
      - V : H_kv * p * d elements
    Total bytes: 2 * p * d * (H_qo + 2 * H_kv)

    Operational intensity:
      I = (4 * H_qo * p^2 * d) / (2 * p * d * (H_qo + 2 * H_kv))
        = (2 * H_qo * p) / (H_qo + 2 * H_kv)

    Args:
        H_qo: Number of query/output heads
        H_kv: Number of key/value heads
        d:    Head dimension
        p:    Prompt (sequence) length

    Returns:
        Operational intensity (FLOPs / byte)
    """
    flops = 4 * H_qo * p * p * d
    bytes_read = 2 * p * d * (H_qo + 2 * H_kv)
    return flops / bytes_read


def decode_op_intensity(H_qo: int, H_kv: int, d: int, c: int) -> float:
    """
    Operational intensity of decode attention.

    At decode time, a single new query token attends over c context tokens.
    For each query head:
      - Q @ K^T : 2 * 1 * c * d = 2 * c * d FLOPs
      - Scores @ V : 2 * 1 * c * d = 2 * c * d FLOPs
    Total FLOPs: 4 * H_qo * c * d

    Memory reads (FP16 => 2 bytes/element):
      - Q : H_qo * 1 * d elements  (1 query token)
      - K : H_kv * c * d elements  (KV cache)
      - V : H_kv * c * d elements  (KV cache)
    Total bytes: 2 * d * (H_qo + 2 * H_kv * c)

    Operational intensity:
      I = (4 * H_qo * c * d) / (2 * d * (H_qo + 2 * H_kv * c))
        = (2 * H_qo * c) / (H_qo + 2 * H_kv * c)

    For large c:  I -> H_qo / H_kv  (the GQA group size G)
    This is small (e.g. 4 for LLaMA3), confirming decode is heavily memory-bound.

    Args:
        H_qo: Number of query/output heads
        H_kv: Number of key/value heads
        d:    Head dimension
        c:    Context length

    Returns:
        Operational intensity (FLOPs / byte)
    """
    flops = 4 * H_qo * c * d
    bytes_read = 2 * d * (H_qo + 2 * H_kv * c)
    return flops / bytes_read


if __name__ == "__main__":
    models = {
        'LLaMA3-1B': llama3_1b_config,
        'LLaMA3-3B': llama3_3b_config,
        'LLaMA3-8B': llama3_8b_config,
    }

    seq_lens  = 2 ** np.arange(7, 16)   # 2^7 to 2^15
    ctx_lens  = 2 ** np.arange(7, 16)   # 2^7 to 2^15

    for name, params in models.items():
        H_qo = params["num_attention_heads"]
        H_kv = params["num_key_value_heads"]
        d    = params["hidden_size"] // H_qo

        print(f"=== {name} (H_qo={H_qo}, H_kv={H_kv}, d={d}) ===")

        print("  Prefill Operational Intensity:")
        for p in seq_lens:
            I = prefill_op_intensity(H_qo, H_kv, d, p)
            print(f"    p={p:6d}: {I:.2f} FLOPs/byte")

        print("  Decode Operational Intensity:")
        for c in ctx_lens:
            I = decode_op_intensity(H_qo, H_kv, d, c)
            print(f"    c={c:6d}: {I:.4f} FLOPs/byte")
        print(f"    (asymptotic limit as c->inf: H_qo/H_kv = {H_qo/H_kv:.1f})")
        print()
