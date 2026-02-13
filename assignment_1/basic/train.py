import torch 
import torch.nn as nn
import numpy.typing as npt
import os 
import typing
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