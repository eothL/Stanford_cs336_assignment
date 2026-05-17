"""Problem (flash_benchmarking): compare Triton FlashAttention-2 vs regular PyTorch attention.

Run on a single H100:
    uv run python -m cs336_systems.flash_benchmarking

Debug a small grid locally first:
    uv run python -m cs336_systems.flash_benchmarking \
        --seq-lens 128 256 --d-heads 64 --dtypes bfloat16
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import subprocess
import sys

import torch

# Faithful to the PDF: seq_len powers of 2 in [128, 65536],
# d_head powers of 2 in [16, 128], precisions {bf16, fp32}.
SEQ_LENS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
D_HEADS = (16, 32, 64, 128)
DTYPES = ("bfloat16", "float32")

BATCH_SIZE = 1          # the problem mandates batch size 1
IS_CAUSAL = True        # the problem mandates causal masking
DEVICE = "cuda"         


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# ───────────────────────── inputs ─────────────────────────
def make_inputs(seq_len: int, d_head: int, dtype: torch.dtype):
    """Random Q, K, V of shape (BATCH_SIZE, seq_len, d_head) on DEVICE.

    3D (no head axis) to match FlashAttention.forward(ctx, Q, K, V, ...);
    requires_grad=True because we also benchmark the backward pass.
    """
    Q = torch.randn((BATCH_SIZE, seq_len, d_head), device = DEVICE, dtype = dtype, requires_grad=True)
    K = torch.randn((BATCH_SIZE, seq_len, d_head), device = DEVICE, dtype = dtype, requires_grad=True)
    V = torch.randn((BATCH_SIZE, seq_len, d_head), device = DEVICE, dtype = dtype, requires_grad=True)

    return Q, K, V



# ───────────────────────── reference attention ─────────────────────────
def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Regular (non-flash) PyTorch attention, causal.

    scaled_dot_product_attention does torch.where(mask, scores, -inf), so
    `mask` is a (queries, keys) bool with True = keep. A lower-triangular
    mask lets query i attend to key j only when j <= i; shape (Sq, Sk)
    broadcasts over the batch dimension.
    """
    from cs336_basics import model

    n_queries, n_keys = q.shape[-2], k.shape[-2]
    causal_mask = torch.tril(
        torch.ones(n_queries, n_keys, dtype=torch.bool, device=q.device)
    )
    return model.scaled_dot_product_attention(q, k, v, mask=causal_mask)

# ───────────────────────── triton flash attention ─────────────────────────
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Triton FlashAttention-2, causal.

    autograd.Function.apply takes positional args only (a kwarg raises
    TypeError); order matches forward(ctx, Q, K, V, is_causal).
    """
    from cs336_systems.flashattention import FlashAttentionTriton as FlashAttention
    return FlashAttention.apply(q, k, v, IS_CAUSAL)


# ───────────────────────── TODO ─────────────────────────
def bench_one(impl, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup:int = 25, rep:int = 100) -> dict[str, float]:
    """Time forward, backward, and end-to-end with triton.testing.do_bench.

    `impl` is one of the (q,k,v) -> o callables above.

    do_bench(fn, warmup=..., rep=..., grad_to_none=[...]) runs `fn` many
    times and returns the median latency in milliseconds. Three closures:

      1. forward only:  run `impl` under torch.no_grad() so no graph is built.

      2. end-to-end:    `impl(q,k,v).sum().backward()`. q/k/v require grad,
                        so pass grad_to_none=[q, k, v] to do_bench, otherwise
                        grads accumulate across reps and pollute the timing.

      3. backward only: THE TRAP. If you build the graph once outside the
                        timed fn and call .backward() repeatedly, the 2nd
                        call dies ("backward through the graph a second
                        time"). Fix: precompute `o = impl(q,k,v)` once, then
                        time `o.sum().backward(retain_graph=True)` with
                        grad_to_none=[q, k, v]. Keep `import triton.testing`
                        local to this function so the file still imports on
                        a triton-less laptop.

    Return: {"fwd_ms": ..., "bwd_ms": ..., "full_ms": ...}
    """
    from triton.testing import do_bench
    def fwd():
        with torch.no_grad():
            impl(q, k, v)

    def full():
        impl(q, k, v).sum().backward() 
    
    fwd_ms = do_bench(fwd, warmup=warmup, rep=rep)
    full_ms = do_bench(full, warmup=warmup, rep=rep, grad_to_none= [q, k, v])
    
    o = impl(q, k, v)
    loss = o.sum()
    def bwd():
        loss.backward(retain_graph=True) # retain_graph to keep the graph through the repetition
    bwd_ms = do_bench(bwd, warmup=warmup, rep=rep,  grad_to_none= [q, k, v])
    return {"fwd_ms":fwd_ms, "full_ms": full_ms, "bwd_ms": bwd_ms}


# ─────────────────────── orchestration ───────────────────────
IMPLS = {"pytorch": pytorch_attention, "flash_triton": flash_attention}
RESULT_PREFIX = "RESULT_JSON "


def bench_single(impl_name: str, dtype_name: str, d_head: int, seq_len: int,
                 warmup: int, rep: int) -> dict:
    """Benchmark exactly one config IN-PROCESS. Returns a result/OOM row.

    Handles the *recoverable* allocator OOM (case 1): catch, reclaim, and
    return an {"oom": True} row. A non-OOM RuntimeError is re-raised so a
    real kernel/shape bug is never silently mislabelled as OOM. The hard
    cases (poisoned CUDA context, OOM-kill) are NOT handled here — that is
    what the subprocess isolation in run_sweep_isolated is for.
    """
    impl = IMPLS[impl_name]
    dtype = _dtype(dtype_name)
    base = {"impl": impl_name, "dtype": dtype_name,
            "d_head": d_head, "seq_len": seq_len}
    q = k = v = None
    try:
        q, k, v = make_inputs(seq_len, d_head, dtype)
        torch.cuda.reset_peak_memory_stats()
        timings = bench_one(impl, q, k, v, warmup, rep)
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        return {**base, **{key: round(value, 4) for key, value in timings.items()},
                "peak_mb": peak_mb}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise  # a genuine bug, not OOM — don't swallow it
        return {**base, "oom": True}
    finally:
        q = k = v = None  # drop the only refs so the graph is collectable
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _format_row(tag: str, row: dict) -> str:
    if row.get("oom"):
        return f"{tag} | OOM"
    return (f"{tag} | fwd={row['fwd_ms']:>9} bwd={row['bwd_ms']:>9} "
            f"full={row['full_ms']:>9} ms | peak={row['peak_mb']} MB")


def _cases(seq_lens, d_heads, dtype_names):
    for impl_name in IMPLS:
        for dtype_name, d_head, seq_len in itertools.product(dtype_names, d_heads, seq_lens):
            yield impl_name, dtype_name, d_head, seq_len


def run_sweep(seq_lens, d_heads, dtype_names, warmup, rep) -> list[dict]:
    """In-process sweep. Survives case-1 OOM only; a poisoned context will
    cascade. Fast for local debugging of small grids."""
    rows: list[dict] = []
    for impl_name, dtype_name, d_head, seq_len in _cases(seq_lens, d_heads, dtype_names):
        tag = f"{impl_name:<13} dtype={dtype_name:<8} d={d_head:<3} seq={seq_len:<6}"
        row = bench_single(impl_name, dtype_name, d_head, seq_len, warmup, rep)
        print(_format_row(tag, row))
        rows.append(row)
    return rows


def _parse_result(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return None


def run_sweep_isolated(seq_lens, d_heads, dtype_names, warmup, rep,
                       timeout: int) -> list[dict]:
    """One subprocess per config. The child's CUDA context dies with it, so
    a poisoned context / OOM-kill / segfault on one cell cannot take down
    the rest of the sweep — the parent just records OOM and moves on."""
    rows: list[dict] = []
    for impl_name, dtype_name, d_head, seq_len in _cases(seq_lens, d_heads, dtype_names):
        tag = f"{impl_name:<13} dtype={dtype_name:<8} d={d_head:<3} seq={seq_len:<6}"
        cmd = [sys.executable, "-m", "cs336_systems.flash_benchmarking",
               "--single", impl_name, dtype_name, str(d_head), str(seq_len),
               "--warmup", str(warmup), "--rep", str(rep)]
        row: dict | None = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            row = _parse_result(proc.stdout)
        except subprocess.TimeoutExpired:
            row = None  # treat a hung config as unusable
        if row is None:
            row = {"impl": impl_name, "dtype": dtype_name,
                   "d_head": d_head, "seq_len": seq_len, "oom": True}
            print(f"{tag} | OOM / crash / timeout (child died)")
        else:
            print(_format_row(tag, row))
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    head = "| impl | dtype | d_head | seq_len | fwd (ms) | bwd (ms) | full (ms) | peak (MB) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        if r.get("oom"):
            lines.append(f"| {r['impl']} | {r['dtype']} | {r['d_head']} | "
                         f"{r['seq_len']} | OOM | OOM | OOM | OOM |")
        else:
            lines.append(f"| {r['impl']} | {r['dtype']} | {r['d_head']} | "
                         f"{r['seq_len']} | {r['fwd_ms']} | {r['bwd_ms']} | "
                         f"{r['full_ms']} | {r['peak_mb']} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="flash_benchmarking: Triton FA-2 vs PyTorch attention")
    p.add_argument("--seq-lens", nargs="+", type=int, default=list(SEQ_LENS))
    p.add_argument("--d-heads", nargs="+", type=int, default=list(D_HEADS))
    p.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup (ms)")
    p.add_argument("--rep", type=int, default=100, help="do_bench rep (ms)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-file", type=str, default=None, help="write the markdown table here")
    p.add_argument("--in-process", action="store_true",
                   help="run the sweep in this process (fast local debugging; "
                        "a poisoned CUDA context will cascade). Default is "
                        "subprocess isolation, one child per config.")
    p.add_argument("--timeout", type=int, default=1200,
                   help="per-config child timeout in seconds (isolated mode)")
    p.add_argument("--single", nargs=4, default=None,
                   metavar=("IMPL", "DTYPE", "D_HEAD", "SEQ_LEN"),
                   help="internal: benchmark one config and print one "
                        "RESULT_JSON line. Used by the isolated runner.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Child mode: benchmark exactly one config, emit one machine-readable
    # line, exit. A recoverable OOM still prints a RESULT_JSON oom row (exit
    # 0); an unrecoverable failure just kills this process and the parent
    # records OOM from the missing line.
    if args.single is not None:
        impl_name, dtype_name, d_head, seq_len = args.single
        row = bench_single(impl_name, dtype_name, int(d_head), int(seq_len),
                           args.warmup, args.rep)
        print(RESULT_PREFIX + json.dumps(row))
        return

    print(f"flash_benchmarking | device={DEVICE} batch={BATCH_SIZE} causal={IS_CAUSAL} "
          f"| mode={'in-process' if args.in_process else 'isolated'}")
    print("=" * 100)

    if args.in_process:
        rows = run_sweep(args.seq_lens, args.d_heads, args.dtypes, args.warmup, args.rep)
    else:
        rows = run_sweep_isolated(args.seq_lens, args.d_heads, args.dtypes,
                                  args.warmup, args.rep, args.timeout)

    print("=" * 100)
    table = format_table(rows)
    print(table)
    if args.results_file:
        with open(args.results_file, "w") as f:
            f.write(table + "\n")
        print(f"\nWrote table to {args.results_file}")


if __name__ == "__main__":
    main()
