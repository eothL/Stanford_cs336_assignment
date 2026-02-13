import torch 
import numpy.typing as npt

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    
    input_sequences = torch.empty((batch_size, context_length), device = device, dtype= torch.long)
    next_token_targets = torch.empty((batch_size, context_length), device = device, dtype= torch.long)

    starters = torch.randint(0, len(x)- context_length, (batch_size,))
    for i in range(batch_size):
        start = starters[i]
        input_sequences[i] = torch.tensor(x[start: start + context_length])
        next_token_targets[i] = torch.tensor(x[start + 1: start + context_length + 1])
        
    return (input_sequences, next_token_targets)
