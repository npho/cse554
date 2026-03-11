from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

import torch

from continous_engine import Engine, Request


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_ids: torch.Tensor
    output_len: int


def generate_request_specs(
    num_requests: int,
    seed: int,
    vocab_size: int,
) -> list[RequestSpec]:
    rng = random.Random(seed)
    specs: list[RequestSpec] = []
    high_token_id = max(101, min(vocab_size - 1, 5000))

    for request_id in range(num_requests):
        prompt_len = rng.randint(1, 10)
        output_len = rng.randint(1, 128)
        prompt_ids = torch.tensor(
            [rng.randint(100, high_token_id) for _ in range(prompt_len)],
            dtype=torch.long,
        )
        specs.append(RequestSpec(request_id, prompt_ids, output_len))

    return specs


def materialize_requests(specs: list[RequestSpec]) -> list[Request]:
    return [
        Request(spec.request_id, spec.prompt_ids.clone(), spec.output_len)
        for spec in specs
    ]


def release_request_cache(engine: Engine, req: Request) -> None:
    cache = engine.kv_cache_map.pop(req.request_id, None)
    if cache is not None:
        cache.release()


def reset_engine_state(engine: Engine) -> None:
    for cache in list(engine.kv_cache_map.values()):
        cache.release()
    engine.kv_cache_map.clear()


def mark_finished(engine: Engine, req: Request, completed: list[Request]) -> None:
    completed.append(req)
    release_request_cache(engine, req)


def run_naive(engine: Engine, specs: list[RequestSpec], batch_size: int) -> list[Request]:
    completed: list[Request] = []
    pending = materialize_requests(specs)

    while pending:
        batch = pending[:batch_size]
        pending = pending[batch_size:]

        first_tokens = engine.run(batch, num_decode_req=0)
        for req, token in zip(batch, first_tokens):
            req.output_token_ids = torch.cat((req.output_token_ids, token.view(1)))

        active_decode: list[Request] = []
        for req in batch:
            generated = req.current_length - req.prompt_length
            if generated >= req.output_length:
                mark_finished(engine, req, completed)
            else:
                active_decode.append(req)

        while active_decode:
            decode_tokens = engine.run(active_decode, num_decode_req=len(active_decode))
            next_active: list[Request] = []

            for req, token in zip(active_decode, decode_tokens):
                req.output_token_ids = torch.cat((req.output_token_ids, token.view(1)))
                generated = req.current_length - req.prompt_length
                if generated >= req.output_length:
                    mark_finished(engine, req, completed)
                else:
                    next_active.append(req)

            active_decode = next_active

    return completed


def run_continuous(engine: Engine, specs: list[RequestSpec], batch_size: int) -> list[Request]:
    completed: list[Request] = []
    pending = materialize_requests(specs)
    decode_req: list[Request] = []

    while pending or decode_req:
        scheduled_prefill: list[Request] = []
        while pending and len(decode_req) + len(scheduled_prefill) < batch_size:
            scheduled_prefill.append(pending.pop(0))

        request_list = decode_req + scheduled_prefill
        if not request_list:
            break

        new_tokens = engine.run(request_list, num_decode_req=len(decode_req))
        for req, token in zip(request_list, new_tokens):
            req.output_token_ids = torch.cat((req.output_token_ids, token.view(1)))

        next_decode: list[Request] = []
        for req in decode_req:
            generated = req.current_length - req.prompt_length
            if generated >= req.output_length:
                mark_finished(engine, req, completed)
            else:
                next_decode.append(req)

        for req in scheduled_prefill:
            generated = req.current_length - req.prompt_length
            if generated >= req.output_length:
                mark_finished(engine, req, completed)
            else:
                next_decode.append(req)

        decode_req = next_decode

    return completed


def time_policy(policy_name: str, runner, engine: Engine, specs: list[RequestSpec], batch_size: int) -> float:
    reset_engine_state(engine)
    torch.cuda.synchronize()
    start = time.perf_counter()
    completed = runner(engine, specs, batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if len(completed) != len(specs):
        raise RuntimeError(
            f"{policy_name} completed {len(completed)} requests, expected {len(specs)}"
        )

    return elapsed

if __name__ == "__main__":
    num_requests = 100
    batch_size = 10
    seed = 1
    trials = 1

    engine = Engine()

    specs = generate_request_specs(
        num_requests=num_requests,
        seed=seed,
        vocab_size=engine.weights["embedding"].size(0),
    )

    warmup_specs = generate_request_specs(
        num_requests=8,
        seed=seed + 1,
        vocab_size=engine.weights["embedding"].size(0),
    )
    time_policy("naive warmup", run_naive, engine, warmup_specs, batch_size=4)
    time_policy("continuous warmup", run_continuous, engine, warmup_specs, batch_size=4)

    naive_times = [
        time_policy("naive", run_naive, engine, specs, batch_size)
        for _ in range(trials)
    ]
    continuous_times = [
        time_policy("continuous", run_continuous, engine, specs, batch_size)
        for _ in range(trials)
    ]

    naive_median = statistics.median(naive_times)
    continuous_median = statistics.median(continuous_times)
    speedup = naive_median / continuous_median

    prompt_lengths = [spec.prompt_ids.numel() for spec in specs]
    output_lengths = [spec.output_len for spec in specs]

    print(f"requests={num_requests}, batch_size={batch_size}, trials={trials}, seed={seed}")
    print(
        "prompt_len_range="
        f"[{min(prompt_lengths)}, {max(prompt_lengths)}], "
        "output_len_range="
        f"[{min(output_lengths)}, {max(output_lengths)}]"
    )
    print(f"naive_times_s={naive_times}")
    print(f"continuous_times_s={continuous_times}")
    print(f"naive_median_s={naive_median:.6f}")
    print(f"continuous_median_s={continuous_median:.6f}")
    print(f"speedup_over_naive={speedup:.4f}x")
