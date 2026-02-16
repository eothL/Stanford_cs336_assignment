from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from .inference import generate_text, load_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained TransformerLM.")
    parser.add_argument("--config", default=None, help="Path to model config (.yaml/.json).")
    parser.add_argument("--vocab-file", default=None, help="Path to vocab JSON.")
    parser.add_argument("--merge-file", default=None, help="Path to merges TXT.")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])

    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint (.pth).")
    parser.add_argument("--prompt", required=True, help="Prompt text to continue from.")
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stop-token", default="<|endoftext|>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)

    # read yaml config 
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**cfg)
    return parser.parse_args()

def decoding(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    config: dict | None,
    prompt: str,
    vocab_path: str | Path | None,
    merge_path: str | Path | None,
    special_tokens: list[str] | None,
    temperature: float,
    stop_token: str,
    max_tokens: int,
    top_p: float | None,
    device: torch.device | None,
    seed: int | None,
) -> dict:
    """    
    return a dict with text, prompt_text, completion_text, token_ids, completion_tokens_ids
    prompt_token_count, completion_token_count and total_token_count
    """
    if config is None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    LM = load_model(config, checkpoint_path, device)

    generation = generate_text(
        LM=LM,
        prompt=prompt,
        merge_path=merge_path,
        vocab_path=vocab_path,
        special_tokens=special_tokens, 
        temperature=temperature,
        stop_token=stop_token,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    
    return generation



def main():
    args = parse_args()

    cfg: dict | None = None
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        raise ValueError("Missing model config. Pass --config path/to/train_config.yaml")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    if cfg.get("d_ff") is None:
        if cfg.get("ff_dimension") is not None:
            cfg["d_ff"] = int(cfg["ff_dimension"])
        elif cfg.get("hidden_dimension") is not None:
            cfg["d_ff"] = 4 * int(cfg["hidden_dimension"])

    generation = decoding(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        config=cfg,
        prompt=args.prompt,
        vocab_path=args.vocab_file,
        merge_path=args.merge_file,
        special_tokens=args.special_tokens,
        temperature=args.temperature,
        stop_token=args.stop_token,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        device=device,
        seed=args.seed,
    )
    print(generation["text"])
    return None


if __name__ == "__main__":
    main()
