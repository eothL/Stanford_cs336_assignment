import argparse
import yaml
import json

def _load_config(path: str| None)->dict:
    if path is None:
        return {}
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        else:
            return yaml.safe_load(f) or {}
        
def parser_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmarking different size model")
    
    # benchmarking parameter
    parser.add_argument("--warmup-step", type=int, default=5, help="Number of warmup step for benchmarking")
    
    # Architecture hyperparameter (default small Transformer config)
    parser.add_argument("--d-model", type=int, default=768, help="model hidden dimension")
    parser.add_argument("--d-ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--vocab-size", type=int, default=10000) # fixed for the homework
    parser.add_argument("--batch-size", type=int, default=4)     # fixed for the homework
    parser.add_argument("--context-length", type=int, default=128, help="context length of the model")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--epoch", type=int, default= 10, help="Number of training step, default = 10")
    
    # benchmarking settings
    parser.add_argument("--rep", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--mode", type=str, default="full", choices=["forward", "full", "train"],
                        help="'forward' = forward only, 'full' = forward + backward, 'train' = full + optimizer step")
    parser.add_argument("--mixed-precision", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--annotate", action="store_true", help="Use NVTX-annotated attention for nsys profiling")
    parser.add_argument("--results-file", type=str, default=None, help="Path to append JSON-lines results (for markdown table generation)")
    parser.add_argument("--memory-profiling", action="store_true", help="Use torch memory profiler")
    # other
    parser.add_argument("--device", type=str, default="cpu", help="device used for training, default: cpu")
    # Read yaml config file
    parser.add_argument("--config", type=str, default=None, help="add path to the yaml config file")
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        cfg = _load_config(pre_args.config)
        parser.set_defaults(**cfg)
    
    return parser.parse_args()