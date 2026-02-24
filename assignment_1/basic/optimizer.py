import torch 
import math
import torch
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional, Any

# Optimizer
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        assert lr > 0
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable]= None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                t = state.get("t", 0) # get iteration number from the state, or initial value
                grad = p.grad.data    # get the gradient of the loss with respect to p
                p.data -= lr/ math.sqrt(t+1) * grad # udpate weight tensor in-place
                state["t"] = t+1  # Increament iteration number
        return loss


@torch.no_grad()
def CautiousWeightDecay(param: Tensor, state: Tensor, lr:float, wd: float)-> None:
    """Apply Cautious Weight decay technique where we use decoupled weight decay only on parameter that share the same direction as their state"""
    mask = (state * param) > 0
    if torch.any(mask):
        param.mul_(1 - lr * wd * mask.to(dtype=param.dtype))


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        cautious_decay: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
        } 
        super().__init__(params, defaults)

    @torch.no_grad()   
    def step(self, closure: Optional[Callable]= None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious_decay = group["cautious_decay"]

            for p in group["params"]:
                grad = p.grad
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha = 1 - beta1) # equivalent to m = beta1 * m + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)# v = beta2 * v + (1 - beta2) * grad * grad
                
                step_size = lr * (math.sqrt(1 - beta2**t) / (1 - (beta1)**t))
                if cautious_decay:
                    CautiousWeightDecay(param=p.data, state=m, lr=lr, wd=wd)
                else:
                    p.data.mul_(1 - lr * wd)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value = -step_size)

        return loss
    

class Muon(torch.optim.Optimizer):
    def __init__(self, params:Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]], lr:float, weight_decay:float, momentum:float, a:float, b:float, c:float, eps:float=1e-8, cautious_decay:bool=False):

        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
            "momentum":momentum,
            "a":a,
            "b":b,
            "c":c,
        } 
        super().__init__(params, defaults)

    @staticmethod
    def _ns_polynomial(mat: Tensor, a: float, b: float, c: float) -> Tensor:
        """
        Shape-aware Newton-Schulz polynomial.
        For tall matrices (rows > cols), use right-side Gram (cols x cols).
        """
        rows, cols = mat.shape
        if rows > cols:
            gram = mat.transpose(-2, -1) @ mat      # (cols, cols)
            gram2 = gram @ gram
            return a * mat + b * (mat @ gram) + c * (mat @ gram2)

        gram = mat @ mat.transpose(-2, -1)          # (rows, rows)
        gram2 = gram @ gram
        return a * mat + b * (gram @ mat) + c * (gram2 @ mat)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            cautious_decay = group["cautious_decay"]
            eps = group["eps"]
            momentum = group["momentum"]
            a = group["a"]
            b = group["b"]
            c = group["c"]

            for p in group["params"]:
                if p.ndim != 2:
                    continue
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_matrix"] = torch.zeros_like(p.data)
                
                state["step"] +=1
                M = state["momentum_matrix"]
                M.mul_(momentum).add_(grad)
                # Keep the momentum buffer as a raw EMA of gradients.
                # Newton-Schulz normalization should be applied on a temporary tensor.
                M_norm = torch.linalg.norm(M)
                if torch.isnan(M_norm) or torch.isinf(M_norm):
                    continue
                X = M / (M_norm + eps)
                O = self._ns_polynomial(X, a=a, b=b, c=c)
                gamma_adj = 0.2 * lr * math.sqrt(max(1,p.data.shape[0]/p.data.shape[1]))
                if cautious_decay:
                    CautiousWeightDecay(param=p.data, state= M, lr=gamma_adj, wd=wd)
                else:
                    p.data.mul_(1 - gamma_adj * wd)
                p.data.add_(-gamma_adj * O)                

        return loss
    

def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float, eps: float = 1e-6,):
    # l2_norm = torch.norm(params)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    total_sq = sum(torch.sum(g*g) for g in grads)
    total_norm = torch.sqrt(total_sq)
    clip_coef = M/(total_norm + eps)
    
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)

    return None