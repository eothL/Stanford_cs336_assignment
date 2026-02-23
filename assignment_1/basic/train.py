import torch 
import torch.nn as nn
from torch import Tensor
import numpy.typing as npt
import os 
import yaml
import json, hashlib
from jaxtyping import Float, Int
import typing
import numpy as np
import argparse
import wandb
import time
from functools import partial
from . import model 

def load_tokens(path, use_memmap:bool):
    if use_memmap:
        return np.memmap(path, dtype=np.uint16, mode="r")
    return np.fromfile(path, dtype=np.uint16)

def data_loading(x: Int[npt.NDArray, "..."], batch_size: int, context_length: int, device: str = "cpu") -> tuple[Int[Tensor, "batch_size seq_len"], Int[Tensor, "batch_size seq_len"]]:
    """load fully dataset to train it"""
    x_t = torch.as_tensor(x, dtype = torch.long, device=device)

    starts = torch.randint(0, len(x_t)- context_length, (batch_size,), device= device) # tensor of size batch with value between 0 and len(x) - context_length
    offsets = torch.arange(context_length, device= device)

    # leveraging broadcast of pytorch to construct the idx tensor
    idx = starts[:, None] + offsets[None,:] # shape from (B,) and (T,) t (B,1) and (1,T)
        
    return (x_t[idx], x_t[idx+1]) # (x_batch, y_batch)

def get_batch(tokens, batch_size, context_length, device):
    max_start = len(tokens) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    
    x = np.stack([tokens[s:s+context_length] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s+1:s+1+context_length] for s in starts]).astype(np.int64)
    # convert to torch tensor 
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    include_optimizer: bool = True,
):
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": iteration,
    }
    if include_optimizer and optimizer is not None:
        if isinstance(optimizer, dict):
            checkpoint["optimizer_state_dict"] = {
                name: opt.state_dict() for name, opt in optimizer.items()
                }
        else:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Add .tmp to partially written checkpoint and replace it if the file is fully saved
    if isinstance(out, (str, os.PathLike)): # replace works only for these types
        out_path = os.fspath(out)
        tmp_path = f"{out_path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, out_path) # rename the file to out_path file name
        return

    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model_state_dict"])

    # some checkpoint doesn't store the optimizer state 
    if "optimizer_state_dict" not in ckpt:
        raise KeyError(
            "optimizer_state_dict missing in checkpoint. "
            "This checkpoint was likely saved with --save-optimizer-state = False."
        )
    elif isinstance(optimizer, dict):
        for name, opt in optimizer.items():
            opt.load_state_dict(ckpt["optimizer_state_dict"][name])
    else:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt["iteration"]

def run_epoch(
        LM: torch.nn.Module, 
        loader,
        loss_fcn,
        optimizers: dict[str, torch.optim.Optimizer] | None,
        max_norm: float,
        lr: float | None = None ,
        lr_scales: dict[str, float] | None = None,
        device: torch.device | None = None,
        training = True,
):
    if training :
        LM.train()
        context = torch.enable_grad()
    else:
        LM.eval()
        context = torch.no_grad()
    
    total_loss = 0.0
    total_sample = 0
    with context:
        x, y = loader()
        logits: Float[Tensor, "batch_size seq_len vocab_size"] = LM(x)
        # flatten matrices with view(-1) and reshape into vocab_size for x
        loss = loss_fcn(predicted_logits= logits.view(-1, logits.size(-1)), targets= y.view(-1))

        if training:
            if optimizers is None:
                raise ValueError("optimizers must be provided when training=True")
            lr_scales = lr_scales or {}

            for name, opt in optimizers.items():
                opt.zero_grad(set_to_none=True) # cleaning grads from previous step
                # updating learning rate 
                scaled_lr = lr if lr is not None else opt.param_groups[0]["lr"]
                scaled_lr *= lr_scales.get(name, 1.0)
                for g in opt.param_groups:
                    g["lr"] = scaled_lr

            loss.backward()
            model.gradient_clipping(LM.parameters(), M = max_norm)
            
            for opt in optimizers.values():
                opt.step()

        batch_size = logits.size(0)
        # if we use loss instead of loss.detach().item(), we will accumulate the tensors in the computation graph
        total_loss += loss.detach().item() * batch_size 
        total_sample += batch_size

    avg_loss = total_loss / total_sample
    return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(
        description= "Train Transformer Language Model"
    )
    # tokenizer
    parser.add_argument("--vocab-file", default="vocab_10k.json")
    parser.add_argument("--merge-file", default="merges_10k.txt")
    parser.add_argument("--base-pattern", default=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    parser.add_argument("--special-tokens", default=["<|endoftext|>"])

    # dataset
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--train-dataset", default="tinystories_train.uint16.bin")
    parser.add_argument("--val-dataset", default="tinystories_val.uint16.bin")
    parser.add_argument("--out-dir", default=None)

    # training config
    parser.add_argument("--use-memmap", action= "store_true") # default False, becomes True if present
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type = int, default = 93)
    parser.add_argument("--run-name", default = "Transformer_LM_from_scratch" )
    parser.add_argument("--run-number",type = int, default = 1)
    parser.add_argument("--save-every", type = int, default = 1000) 
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", type=str, choices= ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], default = "default")
    parser.add_argument("--float32-matmul-precision", type=str, default = "high")
    parser.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=True)    # save optimizer state or not 
    parser.add_argument("--max-token-processed", type= int, default= None)
    parser.add_argument("--max-time", type= float, default= 90.0) # min

    # hyperparameter
    parser.add_argument("--epochs", type= int, default = 100)
    parser.add_argument("--batch-size", type= int, default = 3)
    parser.add_argument("--context-length",type= int, default = 16)
    parser.add_argument("--vocab-size",type= int, default= 10000)
    parser.add_argument("--hidden-dimension",type= int, default= 64)
    parser.add_argument("--ff-dimension", type= int, default = None )
    parser.add_argument("--num-layers",type= int, default= 3)
    parser.add_argument("--num-heads", type= int, default= 4)
    parser.add_argument("--rope-theta", type= float, default= 10000.0)
    
    ## Optimizer
    parser.add_argument(
        "--optimizer-mode",
        type=str,
        choices=["adamw", "muon_adamw"],
        default="muon_adamw",
        help="Choose optimizer setup: all params with AdamW, or split AdamW+Muon.",
    )
    ### AdamW
    parser.add_argument("--betas", nargs= 2, type=float, default=(0.9,0.99))
    parser.add_argument("--weight-decay", type = float, default=1e-2)
    parser.add_argument("--cautious-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-threshold", type=float, default=1.0)
    ### Muon
    parser.add_argument("--a", type=float, default=3.4445)
    parser.add_argument("--b", type=float, default=-4.7750)
    parser.add_argument("--c", type=float, default=2.0315)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--muon-lr-scale", type=float, default=1.0)
    parser.add_argument("--adamw-lr-scale", type=float, default=None, help="AdamW LR multiplier. Default: 0.1 in muon_adamw mode, 1.0 in adamw mode.")

    ## Learning rate scheduler 
    parser.add_argument("--lr", type= float, default= 1e-3) # constant lr
    parser.add_argument("--lr-max", type= float, default=1)
    parser.add_argument("--lr-min", type= float, default=0.01)
    parser.add_argument("--warmup-iters", type= int, default=7)
    parser.add_argument("--cosine-cycle-iters", type= int, default=21)
    
    ## Architecture
    parser.add_argument("--tied-embedding", action=argparse.BooleanOptionalAction, default=False)
    # if mentionned, it will be true and activated
    parser.add_argument("--use-post-norm", action="store_true")
    parser.add_argument("--remove-rope", action="store_true")
    parser.add_argument("--remove-rmsnorm", action="store_true")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--activation-fcn", type=str, default="swiglu", help="Choose your activation function used in FFN in lowercase")

    parser.add_argument("--config", type= str, default= None)
    # read yaml config 
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**cfg)

    return parser.parse_args()

def auto_run_name(args):
    adamw_lr_scale = args.adamw_lr_scale
    if adamw_lr_scale is None:
        adamw_lr_scale = 0.1 if args.optimizer_mode == "muon_adamw" else 1.0

    cfg = {
          "L": args.num_layers,
          "H": args.num_heads,
          "D": args.hidden_dimension,
          "ctx": args.context_length,
          "bs": args.batch_size,
          "lrmax": args.lr_max,
          "wd": args.weight_decay,
          "beta1": args.betas[0],
          "beta2": args.betas[1],
          "m_norm": args.clip_threshold,
          "dataset": args.train_dataset,
          "act_fcn": args.activation_fcn,
          "optimizer_mode": args.optimizer_mode,
          "muon_lr_scale": args.muon_lr_scale,
          "adamw_lr_scale": adamw_lr_scale,
      }
    
    slug = f"L{cfg['L']}-H{cfg['H']}-D{cfg['D']}-ctx{cfg['ctx']}-bs{cfg['bs']}-lr{cfg['lrmax']}-m_norm{cfg['m_norm']}"        
    slug += f"-c_wd{cfg['wd']}" if args.cautious_decay is True else f"-wd{cfg['wd']}" 
    slug += f"-{cfg['act_fcn']}"
    slug += f"-{args.optimizer_mode}"
    if args.optimizer_mode == "muon_adamw":
        if args.muon_lr_scale != 1.0:
            slug += f"-muonlrx{args.muon_lr_scale:g}"
        if adamw_lr_scale != 1.0:
            slug += f"-adamwlrx{adamw_lr_scale:g}"
    elif adamw_lr_scale != 1.0:
        slug += f"-adamwlrx{adamw_lr_scale:g}"

    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{slug}-{h}"


def train():
    # argument
    args = parse_args()
    
    # Parameters
    args.betas = tuple(args.betas) # convert it as a tuple because args. will return a list
    device = torch.device(args.device)
    if device.type == "cuda": 
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        torch.backends.cudnn.allow_tf32 = False # do it for cuDNN ops (mainly convolutions operation)
        # if args.float32_matmul_precision == "highest":
        #    torch.backends.cuda.matmul.allow_tf32 = False this one is redundant as float32==highest already do it
    
    epochs = args.epochs
    batch_size = args.batch_size
    lr_min = args.lr_min
    lr_max = args.lr_max
    warmup = args.warmup_iters
    cosine_cycle = args.cosine_cycle_iters
    max_norm = args.clip_threshold
    max_time = args.max_time # min
    max_time_seconds = float("inf") if max_time <= 0 else 60*max_time # no time limits if it is negative or 0 
    adamw_lr_scale = args.adamw_lr_scale
    if adamw_lr_scale is None:
        adamw_lr_scale = 0.1 if args.optimizer_mode == "muon_adamw" else 1.0

    if args.max_token_processed is not None:
        max_token_processed = args.max_token_processed
        limited_tokens = max_token_processed is not None  
    else:
        max_token_processed = None
        limited_tokens = max_token_processed is not None  

    if args.ff_dimension is None:
        args.ff_dimension = 4 * args.hidden_dimension

    context_length = args.context_length
    if args.train_dataset == "openwebtext_train.uint16.bin":
        dataset_name = "owt"
    else:
        dataset_name = "ts"

    if not args.run_name:
        args.run_name = auto_run_name(args)
    run_name = "_".join([args.run_name, dataset_name]) 
    run_number = args.run_number
    if args.tied_embedding is True:
        run_name ="_".join([run_name, "tied"])
    if args.compile is True:
        run_name = "_".join([run_name, "cpl", args.compile_mode])

    if args.use_qk_norm is True:
        run_name= "_".join([run_name, "qknorm"])

    # file 
    artifacts_folder = "artifacts"
    HERE = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(HERE, artifacts_folder)
    os.makedirs(artifacts_path, exist_ok = True)

    val_data_path = os.path.join(artifacts_path, args.val_dataset)
    train_data_path = os.path.join(artifacts_path, args.train_dataset)

    ## experiment folder
    exp_folder_name = f"experiment_{run_name}"
    exp_path = os.path.join(artifacts_path, exp_folder_name) if args.out_dir is None else args.out_dir
    os.makedirs(exp_path, exist_ok= True)

    wandb.init(
        project = "Transformer_LM_training",
        name = run_name,
        config={"optimizer": args.optimizer_mode, **vars(args)},
    )    
    
    # hyperparameter
    model_cfg = {
        "vocab_size" : args.vocab_size,
        "context_length" : args.context_length,
        "d_model" : args.hidden_dimension,
        "d_ff" : args.ff_dimension,
        "num_layers" : args.num_layers,
        "num_heads" : args.num_heads,
        "rope_theta" : args.rope_theta,
        "tied_embedding": args.tied_embedding,
        "device": device,
        "bias" : args.use_bias,
        "remove_rope" : args.remove_rope,
        "remove_rmsnorm" : args.remove_rmsnorm,
        "use_post_norm" : args.use_post_norm,
        "use_qk_norm" : args.use_qk_norm,
        "activation_fcn": args.activation_fcn
        }

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # model initializing
    LM = model.TransformerLM(**model_cfg).to(device)

    optimizer_bundle: dict[str, torch.optim.Optimizer]
    lr_scales: dict[str, float]
    if args.optimizer_mode == "adamw":
        all_params = [p for p in LM.parameters() if p.requires_grad]
        opt_adamw = model.AdamW(
            all_params,
            lr=args.lr_max,
            betas=args.betas,
            weight_decay=args.weight_decay,
            cautious_decay=args.cautious_decay,
        )
        optimizer_bundle = {"adamw": opt_adamw}
        lr_scales = {"adamw": adamw_lr_scale}
    else:
        muons_params, adamw_params = [], []
        for name, p in LM.named_parameters():
            if not p.requires_grad:
                continue
            use_muon = (
                p.ndim == 2
                and "embedding" not in name
                and "lm_head" not in name
                and not name.startswith("head.")
            )
            (muons_params if use_muon else adamw_params).append(p)

        opt_muon = model.Muon(
            muons_params,
            lr=args.lr_max,
            weight_decay=args.weight_decay,
            cautious_decay=args.cautious_decay,
            a=args.a,
            b=args.b,
            c=args.c,
            momentum=args.momentum,
        )
        opt_adamw = model.AdamW(
            adamw_params,
            lr=args.lr_max,
            betas=args.betas,
            weight_decay=args.weight_decay,
            cautious_decay=args.cautious_decay,
        )
        optimizer_bundle = {"muon": opt_muon, "adamw": opt_adamw}
        lr_scales = {"muon": args.muon_lr_scale, "adamw": adamw_lr_scale}

    loss_fcn = model.cross_entropy

    if args.compile is True:
        LM_compil = torch.compile(LM, mode= args.compile_mode, dynamic= False)
    else:
        LM_compil = LM # no compil but keep the same name for simplicity

    total_params = sum(p.numel() for p in LM.parameters())
    print(f"Model parameters: {total_params}")
    wandb.summary["model_params"] = total_params 

    # loading data
    ## loading token from uint16.bin file
    train_tokens = load_tokens(train_data_path, use_memmap=args.use_memmap)
    val_tokens = load_tokens(val_data_path, use_memmap=args.use_memmap)
    
    if args.use_memmap :
        train_loader = partial(get_batch, train_tokens, batch_size, context_length, device) 
        val_loader = partial(get_batch, val_tokens, batch_size, context_length, device) 
    else:
        train_loader = partial(get_batch, train_tokens, batch_size, context_length, device)
        val_loader = partial(get_batch, val_tokens, batch_size, context_length, device)

    # metrics
    history = []
    best_val = float("inf")
    total_token_processed = 0
    start = time.time()
    accum_time = 0
    epoch = 0
    while epoch < epochs and accum_time < max_time_seconds :
        epoch_start = time.time()
        # forward
        lr = model.learning_rate_schedule(t = epoch, lr_min = lr_min, lr_max = lr_max, Tw = warmup, Tc = cosine_cycle)
        train_loss = run_epoch(
            LM=LM_compil,
            loader=train_loader,
            loss_fcn=loss_fcn,
            max_norm=max_norm,
            optimizers=optimizer_bundle,
            lr=lr,
            lr_scales=lr_scales,
            device=device,
            training=True,
        )
        val_loss = run_epoch(LM=LM, loader=val_loader, loss_fcn=loss_fcn, max_norm=max_norm, optimizers=None, device = device, training = False)
        total_token_processed += batch_size * context_length
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "total_token_processed": total_token_processed})
        
        epoch_time = time.time() - epoch_start

        metrics ={ 
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch_time": epoch_time
            }
        for opt_name, scale in lr_scales.items():
            metrics[f"lr_{opt_name}"] = lr * scale

        if val_loss < best_val:
            best_val = val_loss
            if best_val < 3.5:
                result_path = os.path.join(exp_path, f"result_{run_name}_{run_number}_{epoch}.pth")
                save_checkpoint(
                    model=LM,
                    optimizer=optimizer_bundle,
                    iteration=epoch,
                    out=result_path,
                    include_optimizer=args.save_optimizer_state,
                )
                print(f" New best val {best_val: .4f}. Saved {result_path}")
            metrics["best_val_loss"] = best_val

        if args.save_every and epoch % args.save_every == 0 and epoch != 0:
            result_path = os.path.join(exp_path, f"result_{run_name}_{run_number}_{epoch}.pth")
            checkpoint_path = result_path
            save_checkpoint(
                model=LM,
                optimizer=optimizer_bundle,
                iteration=epoch,
                out=result_path,
                include_optimizer= args.save_optimizer_state,
            )
            print(f"checkpoint saved : { checkpoint_path}")

        wandb.log(metrics)
        accum_time += epoch_time
        epoch += 1
        if limited_tokens and total_token_processed >= max_token_processed:
            break 

    total_minute = (time.time() - start) / 60.0

    print(f"Training complete in { total_minute: .2f} min with the best val loss = {best_val}")
    wandb.summary["total_training_time"] = total_minute
    wandb.finish()

if __name__=="__main__":
    train()


    
