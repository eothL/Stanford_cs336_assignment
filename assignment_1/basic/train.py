import torch 
import torch.nn as nn
import numpy.typing as npt
import os 
import typing
import numpy as np
import argparse
from .Tokenizer import Tokenizer
import model
import wandb

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor(x)

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
    parser.add_argument("--seed", type = int, default= 93)
    parser.add_argument("--run-name")

    # hyperparameter
    parser.add_argument("--epochs", type= int, default = 100)
    parser.add_argument("--batch-size", default = 3)
    parser.add_argument("--context-length", default = 1024)
    parser.add_argument("--vocab-size", default= 10000)
    parser.add_argument("--hidden-dimension", default= 64)
    parser.add_argument("--lr", default= 1e-5)
    parser.add_argument("--num-layers", default= 16)

    # vocab_file = args.vocab_file
    # merge_file = args.merge_file
    # base_pattern = args.base_pattern
    # special_tokens = args.special_tokens
    # dataset = args.dataset
    # device = args.device

    # # hyperparameter
    # epochs = args.epochs
    # batch_size = args.batch_size
    # context_length = args.context_length
    # d_model = args.hidden_dimension
    # num_layers = args.num_layers
    # dtypes = {"Embedding": torch.float32}
    return parser.parse_args()

def train():
    # argument
    args = parse_args()

    if args.seed:
        torch.manual_seed(args.seed)


    # file 
    artifacts_folder = "artifacts"
    HERE = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(HERE, artifacts_folder)
    vocab_path = os.path.join(artifacts_path, vocab_file)
    merge_path = os.path.join(artifacts_path, merge_file)
    data_path = os.path.join(artifacts_path, dataset)

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merge_path,special_tokens=special_tokens) 
    
    # model initializing
    embedding = model.Embedding()
    for layer in 
    

    with open(data_path, "rb") as f:
        data = f.read()
        data_array = np.array(data)

    for i in range(epochs):
        x,y = data_loading(x=data_array, batch_size=batch_size, context_length=context_length, device= device)
        x_embed, y_embed = embedding(x), embedding(y)



    
