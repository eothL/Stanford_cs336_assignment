"""Problem (flash_benchmarking): compare Triton FlashAttention-2 vs regular PyTorch attention.

Run on a single H100:
    uv run python -m cs336_systems.flash_benchmarking

Debug a small grid locally first:
    uv run python -m cs336_systems.flash_benchmarking \
        --seq-lens 128 256 --d-heads 64 --dtypes bfloat16
"""

from __future__ import annotations

import argparse
import itertools

import torch
import time 
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
def bench_one(impl, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
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




# ─────────────────────── orchestration ───────────────────────
def run_sweep(seq_lens, d_heads, dtype_names, warmup, rep) -> list[dict]:
    rows: list[dict] = []
    for impl_name, impl in (("pytorch", pytorch_attention), ("flash_triton", flash_attention)):
        for dtype_name, d_head, seq_len in itertools.product(dtype_names, d_heads, seq_lens):
            dtype = _dtype(dtype_name)
            tag = f"{impl_name:<13} dtype={dtype_name:<8} d={d_head:<3} seq={seq_len:<6}"
            try:
                q, k, v = make_inputs(seq_len, d_head, dtype)
                torch.cuda.reset_peak_memory_stats()
                timings = bench_one(impl, q, k, v)
                peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                row = {
                    "impl": impl_name, "dtype": dtype_name,
                    "d_head": d_head, "seq_len": seq_len,
                    **{key: round(value, 4) for key, value in timings.items()},
                    "peak_mb": peak_mb,
                }
                rows.append(row)
                print(f"{tag} | fwd={row['fwd_ms']:>9} bwd={row['bwd_ms']:>9} "
                      f"full={row['full_ms']:>9} ms | peak={peak_mb} MB")
            except torch.cuda.OutOfMemoryError:
                rows.append({"impl": impl_name, "dtype": dtype_name,
                             "d_head": d_head, "seq_len": seq_len, "oom": True})
                print(f"{tag} | OOM")
                torch.cuda.empty_cache()
            finally:
                del q, k, v
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"flash_benchmarking | device={DEVICE} batch={BATCH_SIZE} causal={IS_CAUSAL}")
    print("=" * 100)

    rows = run_sweep(args.seq_lens, args.d_heads, args.dtypes, args.warmup, args.rep)

    print("=" * 100)
    table = format_table(rows)
    print(table)
    if args.results_file:
        with open(args.results_file, "w") as f:
            f.write(table + "\n")
        print(f"\nWrote table to {args.results_file}")


if __name__ == "__main__":
    main()
