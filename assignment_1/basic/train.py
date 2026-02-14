import torch 
import torch.nn as nn
from torch import Tensor
import numpy.typing as npt
import os 
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

def save_checkpoint(model: torch.nn.Module, optimizer : torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = { 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model : torch.nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"]


class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size,
                d_model,
                device,
                num_layers, 
                num_heads,
                d_ff, 
                context_length, 
                rope_theta, 
                bias, 
                remove_rope,
                remove_rmsnorm,
                use_post_norm,
                ):
        super().__init__()
        self.context_length = context_length
        self.embedding = model.Embedding(num_embeddings= vocab_size, embedding_dim=d_model, device=device, dtype=torch.float32)
        self.transformer_blocks = nn.ModuleList(
            [model.transformer_block(
                d_model = d_model,
                num_heads= num_heads,
                d_ff = d_ff,
                theta = rope_theta,
                max_seq_len= context_length,
                device = device,
                bias= bias,
                remove_rope= remove_rope,
                remove_rmsnorm= remove_rmsnorm,
                use_post_norm=use_post_norm
                ) for _ in range(num_layers)])
        
        self.head = nn.Sequential(
            model.RMSNorm(d_model=d_model, device=device),
            model.Linear(in_features=d_model, out_features=vocab_size, device=device, bias = bias)
        )
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape[1] <= self.context_length

        h = self.embedding(x)
        for block in self.transformer_blocks:
            h = block(h, token_positions = token_positions)
        logits = self.head(h)
        return logits


def run_epoch(
        LM: torch.nn.Module, 
        loader,
        loss_fcn,
        optimizer: torch.optim.Optimizer,
        lr: float | None = None ,
        device: torch.device | None = None,
        training = True,
):
    if training :
        LM.train()
        context = torch.enable_grad()
    else:
        LM.eval()
        context = torch.no_grad()
    
    total_loss = 0
    total_sample = 0
    with context:
        x, y = loader()
        logits: Float[Tensor, "batch_size seq_len vocab_size"] = LM(x)
        # flatten matrices with view(-1) and reshape into vocab_size for x
        loss = loss_fcn(predicted_logits= logits.view(-1, logits.size(-1)), targets= y.view(-1))

        if training:
            optimizer.zero_grad(set_to_none=True) # cleaning grads from previous step
            # updating learning rate 
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss.backward()
            model.gradient_clipping(LM.parameters(), M = 1e-2)
            optimizer.step()

        batch_size = logits.size(0)
        total_loss += loss * batch_size
        total_sample += batch_size

    avg_loss = total_loss/total_sample
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

    parser.add_argument("--train-dataset", default="tinystories_train.uint16.bin")
    parser.add_argument("--val-dataset", default="tinystories_val.uint16.bin")
    parser.add_argument("--out-dir", default=None)
    # default False, becomes True if present
    parser.add_argument("--use-memmap", action= "store_true")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type = int, default = 93)
    parser.add_argument("--run-name", default = "Transformer_LM_from_scratch" )
    parser.add_argument("--run-number",type = int, default = 1)
    parser.add_argument("--save-every", type = int, default = 10)

    # hyperparameter
    parser.add_argument("--epochs", type= int, default = 100)
    parser.add_argument("--batch-size", type= int, default = 3)
    parser.add_argument("--context-length",type= int, default = 16)
    parser.add_argument("--vocab-size",type= int, default= 10000)
    parser.add_argument("--hidden-dimension",type= int, default= 64)
    parser.add_argument("--num-layers",type= int, default= 3)
    parser.add_argument("--num-heads", type= int, default= 4)
    parser.add_argument("--rope-theta", type= float, default= 10000.0)
    
    ## Optimizer
    parser.add_argument("--betas", nargs= 2, type=float, default=(0.9,0.99))
    parser.add_argument("--weight-decay", type = float, default=1e-2)

    ## Learning rate scheduler 
    parser.add_argument("--lr", type= float, default= 1e-3) # constant lr
    parser.add_argument("--lr-max", type= float, default=1)
    parser.add_argument("--lr-min", type= float, default=0.01)
    parser.add_argument("--warmup-iters", type= int, default=7)
    parser.add_argument("--cosine-cycle-iters", type= int, default=21)
    
    ## Architecture
    # if mentionned, it will be true and activated
    parser.add_argument("--use-post-norm", action="store_true")
    parser.add_argument("--remove-rope", action="store_true")
    parser.add_argument("--remove-rmsnorm", action="store_true")
    parser.add_argument("--use-bias", action="store_true")

    return parser.parse_args()

        
def train():
    # argument
    args = parse_args()
    args.betas = tuple(args.betas) # convert it as a tuple because args. will return a list
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    lr_min = args.lr_min
    lr_max = args.lr_max
    warmup = args.warmup_iters
    cosine_cycle = args.cosine_cycle_iters

    context_length = args.context_length
    run_name = args.run_name
    run_number = args.run_number
    # file 
    artifacts_folder = "artifacts"
    HERE = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(HERE, artifacts_folder)
    os.makedirs(artifacts_path, exist_ok = True)

    val_data_path = os.path.join(artifacts_path, args.val_dataset)
    train_data_path = os.path.join(artifacts_path, args.train_dataset)

    ## experiment folder
    exp_folder_name = f"experiment_{run_name}"
    exp_path = os.path.join(artifacts_path,exp_folder_name) if args.out_dir is None else args.out_dir
    os.makedirs(exp_path, exist_ok= True)

    run = wandb.init(
        project = "Transformer_LM_training",
        name = run_name,
        config={
            "optimizer": "AdamW",
            **vars(args),
        },
    )
    
    
    # hyperparameter
    model_cfg = {
        "vocab_size" : args.vocab_size,
        "context_length" : args.context_length,
        "d_model" : args.hidden_dimension,
        "d_ff" : 4 * args.hidden_dimension,
        "num_layers" : args.num_layers,
        "num_heads" : args.num_heads,
        "rope_theta" : args.rope_theta,
        "device": device,
        "bias" : args.use_bias,
        "remove_rope" : args.remove_rope,
        "remove_rmsnorm" : args.remove_rmsnorm,
        "use_post_norm" : args.use_post_norm,
        }

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # model initializing
    LM = TransformerLM(**model_cfg).to(device)
    optimizer = model.AdamW(LM.parameters(), lr = args.lr_max, betas= args.betas, weight_decay=args.weight_decay)
    loss_fcn = model.cross_entropy

    total_params = sum(p.numel() for p in LM.parameters())
    print(f"Model parameters: {total_params}")
    wandb.log({"model_params": total_params})

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
    
    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        # forward
        lr = model.learning_rate_schedule(t = epoch, lr_min = lr_min, lr_max = lr_max, Tw = warmup, Tc = cosine_cycle)
        train_loss = run_epoch(LM=LM, loader=train_loader, loss_fcn=loss_fcn, optimizer=optimizer,lr=lr, device = device, training = True)
        val_loss = run_epoch(LM=LM, loader=val_loader, loss_fcn=loss_fcn, optimizer=optimizer, device = device, training = False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        
        epoch_time = time.time() - epoch_start
        wandb.log(
            { 
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch_time": epoch_time
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            result_path = os.path.join(exp_path, f"result_{run_name}_{run_number}_{epoch}.pth")
            save_checkpoint(model = LM, optimizer = optimizer, iteration = epoch, out = result_path)
            print(f" New best val {best_val: .4f}. Saved {result_path}")
            wandb.log({ "best_val_loss": best_val, "best_epoch": epoch})

        if args.save_every and epoch % args.save_every == 0:
            result_path = os.path.join(exp_path, f"result_{run_name}_{run_number}_{epoch}.pth")
            checkpoint_path = result_path
            save_checkpoint(model = LM, optimizer = optimizer, iteration = epoch, out = result_path)
            print(f"checkpoint saved : { checkpoint_path}")

    total_minute = (time.time() - start) / 60.0

    print(f"Training complete in { total_minute: .2f} min with the best val loss = {best_val}")
    wandb.log({"total_training_time": total_minute})
    wandb.finish()

if __name__=="__main__":
    train()


    
