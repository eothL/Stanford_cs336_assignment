import torch 
import numpy.typing as npt

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor(x)
    input_sequences = torch.empty((batch_size, context_length), device = device, dtype= torch.long)
    next_token_targets = torch.empty((batch_size, context_length), device = device, dtype= torch.long)

    starters = torch.randint(0, len(x)- context_length, (batch_size,)) # tensor of size batch with value between 0 and len(x) - context_length
    for i in range(batch_size):
        start = starters[i]
        input_sequences[i] = x_t[start: start + context_length]
        next_token_targets[i] = x_t[start + 1: start + context_length + 1]
        
    return (input_sequences, next_token_targets)
