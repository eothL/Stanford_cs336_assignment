import torch
from torch import Tensor
from jaxtyping import Float, Int


# Loss function
def cross_entropy(predicted_logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    """
    Substract the largest element for numerical stability
    cancel out log and exp whenever possible 
    Args:
        o_i (float): predicted logits 
        x_i+1 (int): targets, next id token 
    """
    targets = targets.long()
    targets_logits = predicted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = torch.logsumexp(predicted_logits, dim=-1) - targets_logits

    return loss.mean()


# Metrics evaluation
def perplexity(losses: Tensor, m:int)-> Tensor:
    """
    losses log likelihood over all tokens
    m : total number of evaluated tokens
    """
    return torch.exp(sum(losses)/m)
