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