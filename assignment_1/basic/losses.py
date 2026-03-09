import torch
from torch import Tensor
from jaxtyping import Float, Int


# Loss function
def cross_entropy(predicted_logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"], z_loss_coeff: float = 0.0) -> Float[Tensor, ""]:
    """
    Substract the largest element for numerical stability
    cancel out log and exp whenever possible
    Args:
        o_i (float): predicted logits
        x_i+1 (int): targets, next id token
        z_loss_coeff (float): coefficient for z-loss regularization (0.0 = disabled)
    """
    targets = targets.long()
    targets_logits = predicted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(predicted_logits, dim=-1)
    loss = log_z - targets_logits

    loss = loss + z_loss_coeff * log_z.pow(2) if z_loss_coeff > 0.0 else loss
    return loss.mean()


# Metrics evaluation
def perplexity(losses: Tensor, m:int)-> Tensor:
    """
    losses log likelihood over all tokens
    m : total number of evaluated tokens
    """
    return torch.exp(sum(losses)/m)
