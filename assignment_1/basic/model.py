import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from basic.Tokenizer import Tokenizer
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
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,in_features), **factory_kwargs) 
        self.bias = nn.Parameter(torch.empty(out_features,), **factory_kwargs) if bias else None

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
        super.__init__()
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.embedding_layer = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        

    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return token_ids 