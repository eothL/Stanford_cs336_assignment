import torch
import math
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
    

def _adapt_penalty(
    sigma: float,
    primal_residual_norm: float,
    dual_residual_norm: float,
    eta: float,
    omega: float,
    sigma_min: float,
    sigma_max: float,
) -> float:
    """
    Penalty adaptation used in Eq. (4) of the SISA/NSISA paper.
    We keep this as a local analogue (single-process training).
    """
    if primal_residual_norm > omega * dual_residual_norm:
        return max(sigma * (1.0 - eta), sigma_min)
    if dual_residual_norm > omega * primal_residual_norm:
        return min(sigma * (1.0 + eta), sigma_max)
    return sigma


class SISA(torch.optim.Optimizer):
    """
    Single-model adaptation of SISA (arXiv:2502.10784, Alg. 2).

    We keep per-parameter:
    - pi: dual-like accumulator
    - m: second moment of (pi + grad)^2
    - sigma: adaptive penalty
    rho must be > 0
    sigma must be initialized > 0 and is adapted according to the primal/dual residuals as in the original SISA paper.
    beta must be in [0, 1) and controls the momentum on the second moment (m).
    sigma_eta must be in [0, 1) and controls the adaptation rate of sigma.
    sigma_omega must be > 0 and controls the sensitivity of sigma adaptation to the primal/dual residuals.
    weight_decay is applied in a decoupled manner, either cautiously (only on parameters sharing the same direction as their state) or uniformly.
    eps is added to the denominator for numerical stability.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        beta: float = 0.9,
        rho: float = 1.0,
        sigma_init: float = 1.0,
        sigma_eta: float = 0.1,
        sigma_omega: float = 10.0,
        sigma_min: float = 1e-6,
        sigma_max: float = 1e6,
        adaptive_sigma: bool = True,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_decay: bool = False,
    ):
        defaults = {
            "lr": lr,
            "beta": beta,
            "rho": rho,
            "sigma_init": sigma_init,
            "sigma_eta": sigma_eta,
            "sigma_omega": sigma_omega,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "adaptive_sigma": adaptive_sigma,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious_decay = group["cautious_decay"]
            adaptive_sigma = group["adaptive_sigma"]
            sigma_eta = group["sigma_eta"]
            sigma_omega = group["sigma_omega"]
            sigma_min = group["sigma_min"]
            sigma_max = group["sigma_max"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["pi"] = torch.zeros_like(p.data)
                    state["prev_w"] = p.data.detach().clone()
                    state["sigma"] = float(group["sigma_init"])

                state["step"] += 1
                m = state["m"]
                pi = state["pi"]
                sigma = float(state["sigma"])

                direction = pi + grad
                m.mul_(beta).addcmul_(direction, direction, value=1 - beta)
                denom = sigma + rho * torch.sqrt(m) + eps

                if cautious_decay:
                    CautiousWeightDecay(param=p.data, state=direction, lr=lr, wd=wd)
                elif wd > 0:
                    p.data.mul_(1 - lr * wd)

                old_w = p.data.detach().clone()
                p.data.addcdiv_(direction, denom, value=-lr)
                step_delta = p.data - old_w
                pi.add_(step_delta, alpha=sigma)

                if adaptive_sigma:
                    prev_w = state["prev_w"]
                    primal_residual_norm = float(torch.linalg.norm(step_delta))
                    dual_residual_norm = float(sigma * torch.linalg.norm(old_w - prev_w))
                    sigma = _adapt_penalty(
                        sigma=sigma,
                        primal_residual_norm=primal_residual_norm,
                        dual_residual_norm=dual_residual_norm,
                        eta=sigma_eta,
                        omega=sigma_omega,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                    )
                    state["sigma"] = sigma

                state["prev_w"].copy_(old_w)

        return loss


class NSISA(torch.optim.Optimizer):
    """
    Single-model adaptation of NSISA (arXiv:2502.10784, Alg. 3).

    Differences from SISA:
    - Replace raw gradients with Newton-Schulz transformed momentum state.
    - Add epsilon^t perturbation on zero entries for numerical safety.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        beta: float = 0.9,
        momentum: float = 0.95,
        rho: float = 1.0,
        sigma_init: float = 1.0,
        sigma_eta: float = 0.1,
        sigma_omega: float = 10.0,
        sigma_min: float = 1e-6,
        sigma_max: float = 1e6,
        adaptive_sigma: bool = True,
        a: float = 3.4445,
        b: float = -4.7750,
        c: float = 2.0315,
        perturb_eps: float = 1e-8,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_decay: bool = False,
    ):
        defaults = {
            "lr": lr,
            "beta": beta,
            "momentum": momentum,
            "rho": rho,
            "sigma_init": sigma_init,
            "sigma_eta": sigma_eta,
            "sigma_omega": sigma_omega,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "adaptive_sigma": adaptive_sigma,
            "a": a,
            "b": b,
            "c": c,
            "perturb_eps": perturb_eps,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _ns_polynomial(mat: Tensor, a: float, b: float, c: float) -> Tensor:
        rows, cols = mat.shape
        if rows > cols:
            gram = mat.transpose(-2, -1) @ mat
            gram2 = gram @ gram
            return a * mat + b * (mat @ gram) + c * (mat @ gram2)

        gram = mat @ mat.transpose(-2, -1)
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
            beta = group["beta"]
            momentum = group["momentum"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious_decay = group["cautious_decay"]
            adaptive_sigma = group["adaptive_sigma"]
            sigma_eta = group["sigma_eta"]
            sigma_omega = group["sigma_omega"]
            sigma_min = group["sigma_min"]
            sigma_max = group["sigma_max"]
            a = group["a"]
            b = group["b"]
            c = group["c"]
            perturb_eps = group["perturb_eps"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["b"] = torch.zeros_like(p.data)
                    state["pi"] = torch.zeros_like(p.data)
                    state["prev_w"] = p.data.detach().clone()
                    state["sigma"] = float(group["sigma_init"])

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                b_buffer = state["b"]
                pi = state["pi"]
                sigma = float(state["sigma"])

                b_buffer.mul_(momentum).add_(grad)

                # Newton-Schulz projection is matrix-defined.
                # For vectors/scalars we keep the momentum buffer directly.
                if p.ndim >= 2:
                    b_mat = b_buffer if b_buffer.ndim == 2 else b_buffer.reshape(b_buffer.shape[0], -1)
                    b_norm = torch.linalg.norm(b_mat)
                    if torch.isnan(b_norm) or torch.isinf(b_norm):
                        continue
                    x = b_mat / (b_norm + eps)
                    o_mat = self._ns_polynomial(x, a=a, b=b, c=c)
                    o = o_mat.reshape_as(b_buffer)
                else:
                    o = b_buffer

                direction = pi + o
                m.mul_(beta).addcmul_(direction, direction, value=1 - beta)
                denom = sigma + rho * torch.sqrt(m) + eps

                if perturb_eps > 0:
                    zero_mask = direction == 0
                    direction = direction + (perturb_eps**t) * zero_mask.to(dtype=direction.dtype)

                if cautious_decay:
                    CautiousWeightDecay(param=p.data, state=direction, lr=lr, wd=wd)
                elif wd > 0:
                    p.data.mul_(1 - lr * wd)

                old_w = p.data.detach().clone()
                p.data.addcdiv_(direction, denom, value=-lr)
                step_delta = p.data - old_w
                pi.add_(step_delta, alpha=sigma)

                if adaptive_sigma:
                    prev_w = state["prev_w"]
                    primal_residual_norm = float(torch.linalg.norm(step_delta))
                    dual_residual_norm = float(sigma * torch.linalg.norm(old_w - prev_w))
                    sigma = _adapt_penalty(
                        sigma=sigma,
                        primal_residual_norm=primal_residual_norm,
                        dual_residual_norm=dual_residual_norm,
                        eta=sigma_eta,
                        omega=sigma_omega,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                    )
                    state["sigma"] = sigma

                state["prev_w"].copy_(old_w)

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