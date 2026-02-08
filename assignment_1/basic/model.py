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
