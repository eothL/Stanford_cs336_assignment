"""Benchmark PyTorch attention at different scales (section 1.2.1).

Usage:
    uv run python -m cs336_systems.benchmark_attention --device cuda
    uv run python -m cs336_systems.benchmark_attention --device cuda --results-file attn_results.jsonl
"""
import json
import itertools
import timeit
import torch
import numpy as np

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
D_HEADS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]
WARMUP_STEPS = 5
REP = 100


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_attention_compile(d_head, seq_len, device, results_file=None):
    """Benchmark compiled single-head attention for a given (d_head, seq_len) config."""
    compiled = True
    Q = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
    sdpa_compiled = torch.compile(scaled_dot_product_attention)

    # ── warmup (extra for compile — first calls trigger JIT compilation) ──
    for _ in range(WARMUP_STEPS):
        out = sdpa_compiled(Q, K, V)
        loss = out.sum()
        loss.backward()
    _sync(device)

    # ── time forward ──
    fwd_times = []
    for _ in range(REP):
        start = timeit.default_timer()
        out = sdpa_compiled(Q, K, V)
        _sync(device)
        fwd_times.append(timeit.default_timer() - start)

    # ── measure memory before backward ──
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    memory_before_bwd_mb = None
    if device.startswith("cuda"):
        memory_before_bwd_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    # ── time backward ──
    bwd_times = []
    for _ in range(REP):
        # need a fresh forward for each backward
        out = sdpa_compiled(Q, K, V)
        _sync(device)
        start = timeit.default_timer()
        loss = out.sum()
        loss.backward()
        _sync(device)
        bwd_times.append(timeit.default_timer() - start)

    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    result = {
        "compiled": compiled,
        "d_head": d_head,
        "seq_len": seq_len,
        "fwd_mean": round(np.mean(fwd_times), 6),
        "fwd_std": round(np.std(fwd_times), 6),
        "bwd_mean": round(np.mean(bwd_times), 6),
        "bwd_std": round(np.std(bwd_times), 6),
        "memory_before_bwd_mb": memory_before_bwd_mb,
        "peak_memory_mb": peak_memory_mb,
    }

    print(f"[compiled] d_head={d_head:>4}, seq_len={seq_len:>6} | "
          f"fwd: {result['fwd_mean']:.6f}s ± {result['fwd_std']:.6f} | "
          f"bwd: {result['bwd_mean']:.6f}s ± {result['bwd_std']:.6f} | "
          f"mem_before_bwd: {memory_before_bwd_mb} MB | "
          f"peak: {peak_memory_mb} MB")

    if results_file:
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return result

def benchmark_attention(d_head, seq_len, device, results_file=None):
    """Benchmark single-head attention for a given (d_head, seq_len) config."""

    Q = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)

    # ── warmup ──
    for _ in range(WARMUP_STEPS):
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
    _sync(device)

    # ── time forward ──
    fwd_times = []
    for _ in range(REP):
        start = timeit.default_timer()
        out = scaled_dot_product_attention(Q, K, V)
        _sync(device)
        fwd_times.append(timeit.default_timer() - start)

    # ── measure memory before backward ──
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    memory_before_bwd_mb = None
    if device.startswith("cuda"):
        memory_before_bwd_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    # ── time backward ──
    bwd_times = []
    for _ in range(REP):
        # need a fresh forward for each backward
        out = scaled_dot_product_attention(Q, K, V)
        _sync(device)
        start = timeit.default_timer()
        loss = out.sum()
        loss.backward()
        _sync(device)
        bwd_times.append(timeit.default_timer() - start)

    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    result = {
        "d_head": d_head,
        "seq_len": seq_len,
        "fwd_mean": round(np.mean(fwd_times), 6),
        "fwd_std": round(np.std(fwd_times), 6),
        "bwd_mean": round(np.mean(bwd_times), 6),
        "bwd_std": round(np.std(bwd_times), 6),
        "memory_before_bwd_mb": memory_before_bwd_mb,
        "peak_memory_mb": peak_memory_mb,
    }

    print(f"d_head={d_head:>4}, seq_len={seq_len:>6} | "
          f"fwd: {result['fwd_mean']:.6f}s ± {result['fwd_std']:.6f} | "
          f"bwd: {result['bwd_mean']:.6f}s ± {result['bwd_std']:.6f} | "
          f"mem_before_bwd: {memory_before_bwd_mb} MB | "
          f"peak: {peak_memory_mb} MB")

    if results_file:
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.results_file:
        open(args.results_file, "w").close()  # clear file

    print(f"Benchmarking attention | batch={BATCH_SIZE} | device={args.device}")
    print("=" * 100)

    for d_head, seq_len in itertools.product(D_HEADS, SEQ_LENGTHS):
        try:
            benchmark_attention(d_head, seq_len, args.device, args.results_file)
            if args.compile is True:
                benchmark_attention_compile(d_head, seq_len, args.device, args.results_file)
        except torch.cuda.OutOfMemoryError:
            print(f"d_head={d_head:>4}, seq_len={seq_len:>6} | OOM")
            if args.results_file:
                with open(args.results_file, "a") as f:
                    f.write(json.dumps({"d_head": d_head, "seq_len": seq_len, "OOM": True}) + "\n")
            torch.cuda.empty_cache()

    print("=" * 100)
    print("Done.")
