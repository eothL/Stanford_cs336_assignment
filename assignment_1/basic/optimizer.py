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
    

def _sigma_step_update(
    sigma: float,
    gamma: float,
    step: int,
    update_interval: int,
    sigma_min: float,
    sigma_max: float,
) -> float:
    """
    Paper-faithful penalty update backbone:
    sigma_{l+1} = sigma_l / gamma_l
    Optionally applied every k0 steps (supplementary periodic update).
    """
    if update_interval <= 1 or step % update_interval == 0:
        sigma = sigma / gamma
    return min(max(float(sigma), sigma_min), sigma_max)


def _admm_global_update(
    w_local: Tensor,
    pi: Tensor,
    sigma: float,
    lambda_reg: float,
    eps: float,
) -> Tensor:
    """
    One-client analogue of the ADMM global step.
    For lambda_reg=0: w_next = w_local + pi / sigma.
    """
    if lambda_reg > 0:
        return (sigma * w_local + pi) / (sigma + lambda_reg)
    return w_local + pi / (sigma + eps)


class SISA(torch.optim.Optimizer):
    """
    Single-client approximation of SISA (Alg. 2, Eq. 28/29).

    This follows the ADMM-style structure:
    - local step on w_i with preconditioner (sigma + rho * sqrt(m))
    - dual update pi_i
    - global projection update for w

    Notes:
    - `use_internal_lr=True` uses `internal_lr` (initialized from constructor `lr`)
      and ignores externally scheduled `group["lr"]`.
    - `weight_decay` is interpreted as lambda regularization in the global step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        beta: float = 0.9,
        rho: float = 1.0,
        sigma_init: float = 0.01,
        sigma_gamma: float = 1.0,
        sigma_update_interval: int = 1,
        sigma_min: float = 1e-6,
        sigma_max: float = 1e6,
        eta_bound: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_decay: bool = False,
        use_internal_lr: bool = True,
    ):
        defaults = {
            "lr": lr,
            "internal_lr": lr,
            "beta": beta,
            "rho": rho,
            "sigma_init": sigma_init,
            "sigma_gamma": sigma_gamma,
            "sigma_update_interval": sigma_update_interval,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "eta_bound": eta_bound,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
            "use_internal_lr": use_internal_lr,
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
            internal_lr = group.get("internal_lr", lr)
            beta = group["beta"]
            rho = group["rho"]
            eps = group["eps"]
            lambda_reg = group["weight_decay"]
            cautious_decay = group["cautious_decay"]
            use_internal_lr = group["use_internal_lr"]
            sigma_gamma = group["sigma_gamma"]
            sigma_update_interval = group["sigma_update_interval"]
            sigma_min = group["sigma_min"]
            sigma_max = group["sigma_max"]
            eta_bound = group["eta_bound"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["pi"] = torch.zeros_like(p.data)
                    state["w_local"] = p.data.detach().clone()
                    state["sigma"] = float(group["sigma_init"])

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                pi = state["pi"]
                sigma = _sigma_step_update(
                    sigma=float(state["sigma"]),
                    gamma=sigma_gamma,
                    step=step,
                    update_interval=sigma_update_interval,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                )

                direction = pi + grad
                m.mul_(beta).addcmul_(direction, direction, value=1 - beta)
                if eta_bound is not None and eta_bound > 0:
                    m.clamp_(max=eta_bound * eta_bound)
                denom = sigma + rho * torch.sqrt(m) + eps

                step_lr = internal_lr if use_internal_lr else lr
                direction = step_lr * direction

                w_global = p.data
                w_local_new = w_global - direction / denom
                if cautious_decay and lambda_reg > 0:
                    CautiousWeightDecay(
                        param=w_local_new,
                        state=direction,
                        lr=1.0,
                        wd=lambda_reg,
                    )
                pi.add_(w_local_new - w_global, alpha=sigma)
                p.data.copy_(
                    _admm_global_update(
                        w_local=w_local_new,
                        pi=pi,
                        sigma=sigma,
                        lambda_reg=lambda_reg,
                        eps=eps,
                    )
                )

                state["w_local"].copy_(w_local_new)
                state["sigma"] = sigma

        return loss


class NSISA(torch.optim.Optimizer):
    """
    Single-client approximation of NSISA (Alg. 3).

    Differences vs SISA:
    - Newton-Schulz transformed momentum state O_t replaces raw gradient.
    - epsilon^t mask perturbation on zero entries.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        beta: float = 0.9,
        momentum: float = 0.95,
        rho: float = 1.0,
        sigma_init: float = 0.01,
        sigma_gamma: float = 1.0,
        sigma_update_interval: int = 1,
        sigma_min: float = 1e-6,
        sigma_max: float = 1e6,
        eta_bound: float | None = None,
        a: float = 3.4445,
        b: float = -4.7750,
        c: float = 2.0315,
        perturb_eps: float = 1e-8,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_decay: bool = False,
        use_internal_lr: bool = True,
    ):
        defaults = {
            "lr": lr,
            "internal_lr": lr,
            "beta": beta,
            "momentum": momentum,
            "rho": rho,
            "sigma_init": sigma_init,
            "sigma_gamma": sigma_gamma,
            "sigma_update_interval": sigma_update_interval,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "eta_bound": eta_bound,
            "a": a,
            "b": b,
            "c": c,
            "perturb_eps": perturb_eps,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_decay": cautious_decay,
            "use_internal_lr": use_internal_lr,
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
            internal_lr = group.get("internal_lr", lr)
            beta = group["beta"]
            momentum = group["momentum"]
            rho = group["rho"]
            eps = group["eps"]
            lambda_reg = group["weight_decay"]
            cautious_decay = group["cautious_decay"]
            use_internal_lr = group["use_internal_lr"]
            sigma_gamma = group["sigma_gamma"]
            sigma_update_interval = group["sigma_update_interval"]
            sigma_min = group["sigma_min"]
            sigma_max = group["sigma_max"]
            eta_bound = group["eta_bound"]
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
                    state["w_local"] = p.data.detach().clone()
                    state["sigma"] = float(group["sigma_init"])

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                b_buffer = state["b"]
                pi = state["pi"]
                sigma = _sigma_step_update(
                    sigma=float(state["sigma"]),
                    gamma=sigma_gamma,
                    step=step,
                    update_interval=sigma_update_interval,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                )

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
                if eta_bound is not None and eta_bound > 0:
                    m.clamp_(max=eta_bound * eta_bound)
                denom = sigma + rho * torch.sqrt(m) + eps

                if perturb_eps > 0:
                    zero_mask = direction == 0
                    direction = direction + (perturb_eps**step) * zero_mask.to(dtype=direction.dtype)

                step_lr = internal_lr if use_internal_lr else lr
                direction = step_lr * direction

                w_global = p.data
                w_local_new = w_global - direction / denom
                if cautious_decay and lambda_reg > 0:
                    CautiousWeightDecay(
                        param=w_local_new,
                        state=direction,
                        lr=1.0,
                        wd=lambda_reg,
                    )
                pi.add_(w_local_new - w_global, alpha=sigma)
                p.data.copy_(
                    _admm_global_update(
                        w_local=w_local_new,
                        pi=pi,
                        sigma=sigma,
                        lambda_reg=lambda_reg,
                        eps=eps,
                    )
                )

                state["w_local"].copy_(w_local_new)
                state["sigma"] = sigma

        return loss


# Gradient clipping utility
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
