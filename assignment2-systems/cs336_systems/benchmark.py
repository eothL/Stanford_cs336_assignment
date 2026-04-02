import math
import json
import torch
import torch.cuda.nvtx as nvtx
import timeit
import numpy as np
from dataclasses import asdict
from einops import einsum
from contextlib import nullcontext

from cs336_basics import model, nn_utils
from cs336_basics.nn_utils import softmax

from cs336_systems.cli import parser_arg
from cs336_systems.config import ModelConfig


# ── NVTX-annotated attention (for question e) ─────────────────────────
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return output


# ── Core step logic ───────────────────────────────────────────────────
def run_step(LM, x, y, mode, amp_context, optimizer=None):
    """Run a single benchmarking step.

    Modes:
        "forward"  — inference only (no grad)
        "full"     — forward + backward
        "train"    — forward + backward + optimizer.step()
    """
    if mode == "forward":
        LM.eval()
        with torch.no_grad(), amp_context:
            logits = LM(x)
    else:
        LM.train()
        LM.zero_grad(set_to_none=True)
        with amp_context:
            logits = LM(x)
            loss = nn_utils.cross_entropy(logits, y)
        loss.backward()

        if mode == "train" and optimizer is not None:
            optimizer.step()


# ── Timing harness ────────────────────────────────────────────────────
def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmarking_script(LM, x, y, mode, warmup_steps, rep, device, amp_context, optimizer=None):
    """Time `rep` steps after `warmup_steps` warmup, return per-step times."""

    for _ in range(warmup_steps):
        run_step(LM, x, y, mode, amp_context, optimizer)

    _sync(device)

    step_times = []
    for _ in range(rep):
        start = timeit.default_timer()
        run_step(LM, x, y, mode, amp_context, optimizer)
        _sync(device)
        step_times.append(timeit.default_timer() - start)

    return step_times


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parser_arg()

    # optionally monkey-patch attention with NVTX annotations
    if args.annotate:
        model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    device = args.device
    if args.memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    LM = model.BasicsTransformerLM(**asdict(model_config)).to(device)
    if args.compile is True:
        LM = torch.compile(LM)

    # random data
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    # mixed precision context
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.mixed_precision
        else nullcontext()
    )

    # optimizer (only used in "train" mode)
    optimizer = None
    if args.mode == "train":
        optimizer = torch.optim.AdamW(LM.parameters(), lr=1e-4)

    step_times = benchmarking_script(
        LM, x, y,
        mode=args.mode,
        warmup_steps=args.warmup_step,
        rep=args.rep,
        device=device,
        amp_context=amp_context,
        optimizer=optimizer,
    )
    if args.memory_profiling:
        # save then stop recording
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    mean_time = np.mean(step_times)
    std_time = np.std(step_times)

    # peak memory (only meaningful on CUDA)
    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    result = {
        "mode": args.mode,
        "mixed_precision": args.mixed_precision,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ctx_len": args.context_length,
        "mean_time": round(mean_time, 4),
        "std_time": round(std_time, 4),
        "peak_memory_mb": peak_memory_mb,
        "compile": args.compile
    }

    print(f"Mode: {args.mode} | Mixed precision: {args.mixed_precision}")
    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, "
          f"layers={args.num_layers}, heads={args.num_heads}, "
          f"ctx_len={args.context_length}")
    print(f"Mean step time: {mean_time:.4f}s ± {std_time:.4f}s (over {args.rep} steps)")
    if peak_memory_mb is not None:
        print(f"Peak memory: {peak_memory_mb:.2f} MB")

    # append result as JSON line to file
    if args.results_file:
        with open(args.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

def mixed_precision_accumulation():
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype= torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype= torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype= torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype= torch.float16)
        s += x.type(torch.float32)
    print(s)