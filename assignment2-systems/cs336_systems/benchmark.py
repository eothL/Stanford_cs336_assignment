import torch
import timeit
import numpy as np
from contextlib import nullcontext
from dataclasses import asdict

from cs336_basics import model, nn_utils

from cs336_systems.cli import parser_arg
from cs336_systems.config import ModelConfig


def run_step(LM:model.BasicsTransformerLM, x, y, mode):
    """Run a single benchmarking step (forward-only or forward+backward).

    Args:
        LM: the transformer model
        x: input token ids [batch_size, context_length]
        y: target token ids [batch_size, context_length]
        mode: "forward" or "full" (forward + backward)
    """
    if mode == "forward":
        LM.eval()
        context = torch.no_grad()
    else:
        LM.train()
        context = torch.enable_grad()

    with context:
        logits = LM(x)
        
        if mode != "forward":
            LM.zero_grad(set_to_none=True)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
        

    return

def _sync(device):
    """Synchronize GPU if running on CUDA."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmarking_script(LM, x, y, mode, warmup_steps, rep, device):
    """Time `rep` steps after `warmup_steps` warmup, return per-step times."""

    # warmup
    for _ in range(warmup_steps):
        run_step(LM, x, y, mode)

    _sync(device)

    # timed runs — collect per-step times for mean/std
    step_times = []
    for _ in range(rep):
        start = timeit.default_timer()

        run_step(LM, x, y, mode)

        _sync(device)
        step_times.append(timeit.default_timer() - start)

    return step_times


if __name__ == "__main__":
    args = parser_arg()

    # build model from CLI args
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

    # generate random data 
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    step_times = benchmarking_script(
        LM, x, y,
        mode=args.mode,
        warmup_steps=args.warmup_step,
        rep=args.rep,
        device=device
    )

    mean_time = np.mean(step_times)
    std_time = np.std(step_times)
    print(f"Mode: {args.mode} | Mixed precision: {args.mixed_precision}")
    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, "
          f"layers={args.num_layers}, heads={args.num_heads}, "
          f"ctx_len={args.context_length}")
    print(f"Mean step time: {mean_time:.4f}s ± {std_time:.4f}s (over {args.rep} steps)")
