import math
import json
import torch
import torch.cuda.nvtx as nvtx
import timeit
import numpy as np
from dataclasses import asdict
from einops import einsum

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
def run_step(LM, x, y, mode, optimizer=None):
    """Run a single benchmarking step.

    Modes:
        "forward"  — inference only (no grad)
        "full"     — forward + backward
        "train"    — forward + backward + optimizer.step()
    """
    if mode == "forward":
        LM.eval()
        with torch.no_grad():
            logits = LM(x)
    else:
        LM.train()
        LM.zero_grad(set_to_none=True)
        logits = LM(x)
        loss = nn_utils.cross_entropy(logits, y)
        loss.backward()

        if mode == "train" and optimizer is not None:
            optimizer.step()


# ── Timing harness ────────────────────────────────────────────────────
def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmarking_script(LM, x, y, mode, warmup_steps, rep, device, optimizer=None):
    """Time `rep` steps after `warmup_steps` warmup, return per-step times."""

    for _ in range(warmup_steps):
        run_step(LM, x, y, mode, optimizer)

    _sync(device)

    step_times = []
    for _ in range(rep):
        start = timeit.default_timer()
        run_step(LM, x, y, mode, optimizer)
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
    LM = model.BasicsTransformerLM(**asdict(model_config)).to(device)

    # random data
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

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
        optimizer=optimizer,
    )

    mean_time = np.mean(step_times)
    std_time = np.std(step_times)

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
    }

    print(f"Mode: {args.mode} | Mixed precision: {args.mixed_precision}")
    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, "
          f"layers={args.num_layers}, heads={args.num_heads}, "
          f"ctx_len={args.context_length}")
    print(f"Mean step time: {mean_time:.4f}s ± {std_time:.4f}s (over {args.rep} steps)")

    # append result as JSON line to file
    if args.results_file:
        with open(args.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
