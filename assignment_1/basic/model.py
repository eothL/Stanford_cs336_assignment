import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):
    weight : Float[Tensor, "out_features in_features"]
    bias : Float[Tensor, "out_features"] | None
    def __init__(self, 
                  in_features: int,
                  out_features: int, 
                  device = None, 
                  dtype = None, 
                  bias: bool = True):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features,in_features), **self.factory_kwargs)) 
        self.bias = nn.Parameter(torch.empty((out_features,), **self.factory_kwargs)) if bias else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return x@self.weight.T + self.bias
        else:
            return x@self.weight.T


class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings:int, # size of the vocabulary 
                 embedding_dim: int, # dimension of the embedding vector ie d_model
                 device = None, 
                 dtype=None):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype
        self.num_embeddings =num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **self.factory_kwargs))
        
        nn.init.trunc_normal_(self.weight) # fill the matrix from truncated normal distribution between -3 sigma and 3 sigma
    
    def forward(self, token_ids: Int[Tensor, "..."]) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps:float = 1e-5, device= None, dtype=None):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype

        self.d_model =d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,),**self.factory_kwargs))
        
        nn.init.trunc_normal_(self.weight)

    def forward(self, x:Float[Tensor, "batch sequence length d_model"])-> torch.Tensor:
        # prevent overflow when applying square to input convert input to float 32
        in_dtype = x.dtype 
        x_fp32 =x.to(torch.float32)
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        RMS = torch.rsqrt(mean_square + self.eps) #rsqrt is reverser sqrt 1/sqrt(X)
        result = x_fp32*self.weight*RMS
        return result.to(in_dtype) 
    

class positionwise_feedforward(nn.Module):
    w1_weight : Float[Tensor, " d_ff d_model"]
    w3_weight : Float[Tensor, "d_ff d_model"]
    w2_weight : Float[Tensor, "d_model d_ff"]

    def __init__(self, d_model: int, d_ff: int | None = None, device= None, dtype = None):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(((8/3 * d_model)//64)*64) # keep a multiple of 64 to make a good use of the hardware
        self.w1_weight = nn.Parameter(torch.empty((self.d_ff, self.d_model), **self.factory_kwargs))
        self.w3_weight = nn.Parameter(torch.empty((self.d_ff, self.d_model), **self.factory_kwargs))
        self.w2_weight = nn.Parameter(torch.empty((self.d_model, self.d_ff), **self.factory_kwargs))

        nn.init.trunc_normal_(self.w1_weight)
        nn.init.trunc_normal_(self.w2_weight)
        nn.init.trunc_normal_(self.w3_weight)
    
    @staticmethod
    def SiLU(x:Float[Tensor, "..."])-> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)
    
    def SwiGLU(self,x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_ff"]:
        # x is often a row vector in Pytorch
        # instead of doing W1@x for column vector we need to do x@W1.T
        # elementwise multiplication
        return torch.mul( 
            self.SiLU(x @ self.w1_weight.T), 
            x @ self.w3_weight.T
            )
    
    def forward(self,x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        return self.SwiGLU(x)@self.w2_weight.T
    

class RoPE_full_matrix(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k% 2 ==0 

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        k_idx = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        inv_freq = self.theta**(-2*k_idx/self.d_k)
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angle = pos[:, None] * inv_freq[None,:]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        R = torch.zeros((max_seq_len,d_k,d_k), device=device, dtype=torch.float32)
        even = 2 * torch.arange(self.d_k // 2, device=device)
        odd = even + 1

        R[:, even, odd] = -sin
        R[:, odd, even] = sin
        R[:, even, even] = cos
        R[:, odd, odd] = cos
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        # Indices for tensor lookup must be integer type (int64/long in PyTorch)
        token_positions = token_positions.to(torch.long)
        R_i =self.R[token_positions] #(..., seq_len, d_k, d_k)
        y= R_i @ x.unsqueeze(-1) # (..., seq_len, d_k, d_k) * (..., seq_len, d_k, 1)
        return y.squeeze(-1) #(..., seq_len, d_k)

class RoPE(nn.Module):
    # theta: value for the RoPE
    # d_k: dimension of query and key vectors
    # maximum sequence length that will be inputted
    # device to store the buffer on
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k% 2 ==0 
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(d_k // 2, device=device, dtype=torch.float32)            #(d_k//2,)
        inv_freq = theta ** (-2.0 * k / d_k)                                      #(d_k//2,)
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)       #(max_seq_len,)
        angles = pos[:, None] * inv_freq[None, :]                                 #(max_seq_len, d_k//2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        # Ensure positions are int64 so we can index into the cached (max_seq_len, d_k//2) cos/sin tables
        token_positions = token_positions.to(torch.long) # new tensor with dtype = int64 if it is not already the case else return the same tensor

        cos = self.cos_cached[token_positions]   # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]   # (..., seq_len, d_k//2)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class Softmax(nn.Module):
    # d_i : a dimension i and apply softmax to the i-th dimension of the input tensor
    # For numerical stability, we will substract the largest value in the input tensor as softmax operation is invariant to adding any constant c to all inputs
    def __init__(self, d_i: int):
        super().__init__()
        self.d_i = d_i

    def forward(self, x:Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        exp_x_stable = torch.exp(x - x.amax(dim= self.d_i, keepdim=True))
        return exp_x_stable/exp_x_stable.sum(dim= self.d_i, keepdim=True)
    

