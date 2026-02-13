import torch 
import numpy.typing as npt

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor(x)

    starts = torch.randint(0, len(x_t)- context_length, (batch_size,), device= device) # tensor of size batch with value between 0 and len(x) - context_length
    offsets = torch.arange(context_length, device= device)

    idx = starts[:, None] + offsets[None,:]
        
    return (x_t[idx], x_t[idx+1])
