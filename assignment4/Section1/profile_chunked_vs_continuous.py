"""Profile chunked prefill vs continuous batching.

Input lengths: lognormal(mu=6, sigma=0.7)  → typically ~400-600 tokens
Output lengths: uniform [1, 512]
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

from continous_engine import Engine as ContinuousEngine
from chunked_engine import Engine as ChunkedEngine
from chunked_scheduler import Scheduler as ChunkedScheduler, InputRequest


# ---------------------------------------------------------------------------
# Request specification (shared across policies)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RequestSpec:
    prompt_ids: torch.Tensor
    output_len: int


def generate_specs(
    num_requests: int,
    seed: int,
    tokenizer,
) -> list[RequestSpec]:
    rng = np.random.default_rng(seed)

    # input lengths ~ LogNormal(mu=6, sigma=0.7)
    raw_lens = rng.lognormal(mean=6.0, sigma=0.7, size=num_requests)
    prompt_lens = np.clip(raw_lens.astype(int), 1, 2048).tolist()

    # output lengths ~ Uniform[1, 512]
    output_lens = rng.integers(1, 513, size=num_requests).tolist()

    # Build prompts as repeated tokens (avoids real text)
    vocab_size = tokenizer.vocab_size
    token_ids = rng.integers(100, min(vocab_size, 5000), size=max(prompt_lens))

    specs: list[RequestSpec] = []
    for pl, ol in zip(prompt_lens, output_lens):
        ids = torch.tensor(token_ids[:pl], dtype=torch.long)
        specs.append(RequestSpec(prompt_ids=ids, output_len=int(ol)))
    return specs


# ---------------------------------------------------------------------------
# Utilities shared between runners
# ---------------------------------------------------------------------------
def _release(engine, req_id: int) -> None:
    cache = engine.kv_cache_map.pop(req_id, None)
    if cache is not None:
        cache.release()


def reset_engine(engine) -> None:
    for cache in list(engine.kv_cache_map.values()):
        cache.release()
    engine.kv_cache_map.clear()


# ---------------------------------------------------------------------------
# Continuous batching runner (mirrors profile_naive_vs_continuous.py)
# ---------------------------------------------------------------------------
def run_continuous(engine, specs, batch_size):
    from continous_scheduler import Scheduler, InputRequest
    scheduler = Scheduler(engine, req_batch_size=batch_size)
    for i, spec in enumerate(specs):
        text = engine.tokenizer.decode(spec.prompt_ids.tolist(), skip_special_tokens=True)
        scheduler.add_req(InputRequest(text, spec.output_len))
    while not scheduler.finished():
        scheduler.run()
    for req in scheduler.completed:
        _release(engine, req.request_id)
    return len(scheduler.completed)

# ---------------------------------------------------------------------------
# Chunked-prefill runner (via Scheduler)
# ---------------------------------------------------------------------------
def run_chunked(
    engine: ChunkedEngine,
    specs: list[RequestSpec],
    token_batch_size: int,
) -> int:
    scheduler = ChunkedScheduler(engine, token_batch_size=token_batch_size)

    for i, spec in enumerate(specs):
        text = engine.tokenizer.decode(spec.prompt_ids.tolist(), skip_special_tokens=True)
        scheduler.add_req(InputRequest(text, spec.output_len))

    while not scheduler.finished():
        scheduler.run()

    # release KV caches for completed requests
    for req in scheduler.completed:
        _release(engine, req.request_id)

    return len(scheduler.completed)


# ---------------------------------------------------------------------------
# Timed wrapper
# ---------------------------------------------------------------------------
def time_run(name: str, fn, engine) -> float:
    reset_engine(engine)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    completed = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] completed={completed}  time={elapsed:.3f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_REQUESTS = 256
    CONTINUOUS_BATCH_SIZE = 64       # max concurrent requests for continuous batching
    CHUNKED_TOKEN_BATCH_SIZE = 1024  # max tokens per step for chunked prefill
    SEED = 42
    TRIALS = 1

    # ---- Continuous batching phase ----------------------------------------
    print("Loading continuous engine...")
    c_engine = ContinuousEngine()

    print(f"\nGenerating {NUM_REQUESTS} requests (lognormal inputs, uniform outputs)...")
    specs = generate_specs(NUM_REQUESTS, SEED, c_engine.tokenizer)

    prompt_lens = [s.prompt_ids.numel() for s in specs]
    output_lens = [s.output_len for s in specs]
    print(f"  prompt_len: min={min(prompt_lens)} max={max(prompt_lens)} "
          f"mean={sum(prompt_lens)/len(prompt_lens):.1f}")
    print(f"  output_len: min={min(output_lens)} max={max(output_lens)} "
          f"mean={sum(output_lens)/len(output_lens):.1f}")

    print("\nWarm-up (continuous)...")
    warmup_specs = generate_specs(8, SEED + 1, c_engine.tokenizer)
    time_run("continuous warmup", lambda: run_continuous(c_engine, warmup_specs, 8), c_engine)

    print(f"\nBenchmark continuous ({TRIALS} trial(s))...")
    continuous_times: list[float] = []
    for trial in range(TRIALS):
        ct = time_run(
            f"continuous trial {trial+1}",
            lambda: run_continuous(c_engine, specs, CONTINUOUS_BATCH_SIZE),
            c_engine,
        )
        continuous_times.append(ct)

    # Free GPU memory before loading chunked engine
    reset_engine(c_engine)
    del c_engine
    torch.cuda.empty_cache()

    # ---- Chunked prefill phase --------------------------------------------
    print("\nLoading chunked engine...")
    k_engine = ChunkedEngine()

    print("\nWarm-up (chunked)...")
    warmup_specs_k = generate_specs(8, SEED + 1, k_engine.tokenizer)
    time_run("chunked warmup", lambda: run_chunked(k_engine, warmup_specs_k, 256), k_engine)

    print(f"\nBenchmark chunked ({TRIALS} trial(s))...")
    chunked_times: list[float] = []
    for trial in range(TRIALS):
        kt = time_run(
            f"chunked   trial {trial+1}",
            lambda: run_chunked(k_engine, specs, CHUNKED_TOKEN_BATCH_SIZE),
            k_engine,
        )
        chunked_times.append(kt)

    cont_median = statistics.median(continuous_times)
    chun_median = statistics.median(chunked_times)
    speedup = cont_median / chun_median

    print("\n=== Results ===")
    print(f"num_requests={NUM_REQUESTS}  "
          f"continuous_batch_size={CONTINUOUS_BATCH_SIZE}  "
          f"chunked_token_batch_size={CHUNKED_TOKEN_BATCH_SIZE}")
    print(f"continuous_times_s = {[f'{t:.3f}' for t in continuous_times]}")
    print(f"chunked_times_s    = {[f'{t:.3f}' for t in chunked_times]}")
    print(f"continuous_median  = {cont_median:.3f}s")
    print(f"chunked_median     = {chun_median:.3f}s")
    if speedup >= 1.0:
        print(f"chunked is {speedup:.3f}x faster than continuous batching")
    else:
        print(f"continuous is {1/speedup:.3f}x faster than chunked prefill")
