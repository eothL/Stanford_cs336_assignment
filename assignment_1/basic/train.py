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
from .Tokenizer import Tokenizer
from . import model 

def data_loading(x: Int[npt.NDArray, "..."], batch_size: int, context_length: int, device: str = "cpu") -> tuple[Int[Tensor, "batch_size seq_len"], Int[Tensor, "batch_size seq_len"]]:
    x_t = torch.as_tensor(x, dtype = torch.long, device=device)

    starts = torch.randint(0, len(x_t)- context_length, (batch_size,), device= device) # tensor of size batch with value between 0 and len(x) - context_length
    offsets = torch.arange(context_length, device= device)

    # leveraging broadcast of pytorch to construct the idx tensor
    idx = starts[:, None] + offsets[None,:] # shape from (B,) and (T,) t (B,1) and (1,T)
        
    return (x_t[idx], x_t[idx+1]) # (x_batch, y_batch)


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


def run_epoch(
        LM: torch.nn.Module, 
        loader,
        loss_fcn,
        optimizer: torch.optim.Optimizer,
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
        loss = model.cross_entropy(predicted_logits= logits.view(-1, logits.size(-1)), targets= y.view(-1))

        if training:
            optimizer.zero_grad(set_to_none=True) # cleaning grads from previous step
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
    parser.add_argument("--vocab-file", default="vocab_10k.json")
    parser.add_argument("--merge-file", default="merges_10k.txt")
    parser.add_argument("--base-pattern", default=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    parser.add_argument("--special-tokens", default=["<|endoftext|>"])
    parser.add_argument("--dataset", default="tinystories_val.uint16.bin")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type = int, default = 93)
    parser.add_argument("--run-name", default = "Transformer_LM_from_scratch" )
    parser.add_argument("--run-number",type = int, default = 1)
    parser.add_argument("--save-every", type = int, default = 10)

    # hyperparameter
    parser.add_argument("--epochs", type= int, default = 100)
    parser.add_argument("--batch-size", default = 3)
    parser.add_argument("--context-length", default = 16)
    parser.add_argument("--vocab-size", default= 10000)
    parser.add_argument("--hidden-dimension", default= 64)
    parser.add_argument("--lr", default= 1e-5)
    parser.add_argument("--num-layers", default= 3)
    parser.add_argument("--num-heads", type= int, default= 4)
    parser.add_argument("--rope-theta", type= float, default= 10000.0)
    parser.add_argument("--lr-max", type= float, default=1)
    parser.add_argument("--lr-min", type= float, default=0.01)
    parser.add_argument("--warmup-iters", type= int, default=7)
    parser.add_argument("--cosine-cycle-iters", type= int, default=21)

    # Architecture
    parser.add_argument("--use-post-norm", type= bool, default=False)
    parser.add_argument("--remove-rope", type= bool, default=False)
    parser.add_argument("--remove-rmsnorm", type= bool, default=False)
    parser.add_argument("--use-bias", type = bool, default= False)

    return parser.parse_args()

        
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


def train():
    # argument
    args = parse_args()
    base_pattern = args.base_pattern
    special_tokens = args.special_tokens
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

    vocab_path = os.path.join(artifacts_path, args.vocab_file)
    merge_path = os.path.join(artifacts_path, args.merge_file)
    data_path = os.path.join(artifacts_path, args.dataset)

    ## experiment folder
    exp_folder_name = f"experiment_{run_name}"
    exp_path = os.path.join(artifacts_path,exp_folder_name)
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

    total_params = sum(p.numel() for p in LM.parameters)
    print(f"Model parameters: {total_params}")
    wandb.log({"model_params": total_params})

    with open(data_path, "rb") as f:
        data = f.read()
        data_array = np.array(data)

    # metrics
    history = []
    best_val = float("inf")

    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        # forward
        x,y = data_loading(x=data_array, batch_size=batch_size, context_length=context_length, device= device)
        lr = model.learning_rate_schedule(t = epoch, lr_min = lr_min, lr_max = lr_max, Tw = warmup, Tc = cosine_cycle)
        optimizer = model.AdamW(LM.parameters(), lr = lr)
        train_loss = run_epoch(LM=LM, loader=(x,y), loss_fcn=model.cross_entropy, optimizer=optimizer, device = device, training = True)
        val_loss = run_epoch(LM=LM, loader=(x,y), loss_fcn=model.cross_entropy, optimizer=optimizer, device = device, training = False)
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
            save_checkpoint
            print(f"checkpoint saved : { checkpoint_path}")

    total_minute = (time.time() - start) / 60.0

    print(f"Training complete in { total_minute: .2f} min with the best val loss = {best_val}")
    wandb.log({"total_training_time": total_minute})
    wandb.finish()

if __name__=="__main__":
    train()


    
