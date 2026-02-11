import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import math


class Linear(nn.Module):
    """ 
    in_feature: final dimension of the input
    out_features: inal dimension of the output
    device: torchdevice to store the parameters on 
    dtype Data type of the parameter
    bias add bias parameter or not 
    """
    weight : Float[Tensor, "out_features in_features"]
    bias : Float[Tensor, "out_features"] | None
    def __init__(self, 
                  in_features: int,
                  out_features: int, 
                  device : torch.device | None = None, 
                  dtype : torch.dtype | None = None, 
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
        self.bias = nn.Parameter(torch.empty((out_features,), **self.factory_kwargs)) if bias is True else None

        nn.init.trunc_normal_(self.weight)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias) 

    def forward(self, x:Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        if self.bias is not None:
            return x@self.weight.T + self.bias 
        else:
            return x@self.weight.T


class Embedding(nn.Module):
    """
    Args:
        num_embeddings (int): size of the vocabulary 
        embedding_dim (int): dimension of the embedding vector ie d_model
        device (torch.device|None): Device to store the parameters on  
        dtype (torch.dtype|None): Data type of the parameters
    """
    def __init__(self, 
                 num_embeddings:int,
                 embedding_dim: int,
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
    """
    Args:
        d_model (int): Hidden dimension of the model
        eps (float): Epsilon value for stability
        device (torch.device|None): Device to store the parameters on  
        dtype (torch.dtype|None): Data type of the parameters
    """
    def __init__(self, d_model: int, eps:float = 1e-5, device= None, dtype=None):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype

        self.d_model =d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.empty((d_model,),**self.factory_kwargs))
        
        nn.init.trunc_normal_(self.weights)

    def forward(self, x:Float[Tensor, "... d_model"])-> torch.Tensor:
        # prevent overflow when applying square to input convert input to float 32
        in_dtype = x.dtype 
        x_fp32 =x.to(torch.float32)
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        RMS = torch.rsqrt(mean_square + self.eps) #rsqrt is reverser sqrt 1/sqrt(X)
        result = x_fp32*self.weights*RMS
        return result.to(in_dtype) 
    

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device= None, dtype = None, bias = False):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(((8/3 * d_model)//64)*64) # keep a multiple of 64 to make a good use of the hardware
        self.w1_proj = Linear(self.d_model, self.d_ff,  **self.factory_kwargs, bias=bias)
        self.w3_proj = Linear(self.d_model, self.d_ff,  **self.factory_kwargs, bias=bias)
        self.w2_proj = Linear(self.d_ff, self.d_model,  **self.factory_kwargs, bias=bias)

        nn.init.trunc_normal_(self.w1_proj.weight)
        nn.init.trunc_normal_(self.w2_proj.weight)
        nn.init.trunc_normal_(self.w3_proj.weight)
    
    @staticmethod
    def SiLU(x:Float[Tensor, "..."])-> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)
    
    def SwiGLU(self,x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_ff"]:
        # x is often a row vector in Pytorch
        # instead of doing W1@x for column vector we need to do x@W1.T
        # elementwise multiplication
        return torch.mul( 
            self.SiLU(self.w1_proj(x)), 
            self.w3_proj(x)
            )
    
    def forward(self,x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        return self.w2_proj(self.SwiGLU(x))
        

class RoPE_full_matrix(nn.Module):
    """
    Args:
        theta (float): Angle value for the RoPE
        d_k (int): dimension of query and key vector
        max_seq_len (int): maximum sequence length that will be inputted
        device (torch.device|None): Device to store the buffer on
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
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
        assert x.shape[:-2] == token_positions.shape[::]
        # Indices for tensor lookup must be integer type (int64/long in PyTorch)
        token_positions = token_positions.to(torch.long)
        R_i =self.R[token_positions] #(..., seq_len, d_k, d_k)
        y= R_i @ x.unsqueeze(-1) # (..., seq_len, d_k, d_k) * (..., seq_len, d_k, 1)
        return y.squeeze(-1) #(..., seq_len, d_k)

class RoPE(nn.Module):
    """
    Args:
        theta (float): Angle value for the RoPE
        d_k (int): dimension of query and key vector
        max_seq_len (int): maximum sequence length that will be inputted
        device (torch.device|None): Device to store the buffer on
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k% 2 ==0 
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k:Float[Tensor, "d_k_half"] = torch.arange(d_k // 2, device=device, dtype=torch.float32)  
        inv_freq:Float[Tensor, "d_k_half"]= theta ** (-2.0 * k / d_k)                                     
        pos:Float[Tensor, "max_seq_len"] = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles:Float[Tensor, "max_seq_len d_k_half"] = pos[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x:Float[Tensor, "... seq_len d_k"], token_positions:Int[Tensor, "... seq_len"])-> Float[Tensor, "... seq_len d_k"]:
        # Ensure positions are int64 so we can index into the cached (max_seq_len, d_k_half) cos/sin tables
        token_positions = token_positions.to(torch.long) # new tensor with dtype = int64 if it is not already the case else return the same tensor

        cos: Float[Tensor, "... seq_len d_k_half"] = self.cos_cached[token_positions]
        sin: Float[Tensor, "... seq_len d_k_half"] = self.sin_cached[token_positions]

        x_even: Float[Tensor, "... seq_len d_k_half"] = x[..., 0::2]
        x_odd: Float[Tensor, "... seq_len d_k_half"]  = x[..., 1::2]

        out: Float[Tensor, "... seq_len d_k"] = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out # (... seq_len d_k)


class Softmax(nn.Module):
    """
    Args:
        d_i (int): a dimension i and apply softmax to the i-th dimension of the input tensor
    For numerical stability, we will substract the largest value in the input tensor as softmax operation is invariant to adding any constant c to all inputs
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.d_i = dim

    def forward(self, x:Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        exp_x_stable = torch.exp(x - x.amax(dim= self.d_i, keepdim=True))
        return exp_x_stable/exp_x_stable.sum(dim= self.d_i, keepdim=True)
    

class scaled_dot_product_attention(nn.Module):
    def __init__(self, mask: Float[Tensor, "seq_len seq_len"] | None = None, device: torch.device | None = None):
        super().__init__()
        self.mask = mask.to(device=device) if mask is not None else None

    def forward(self, Q:Float[Tensor, "... seq_len d_k"], K:Float[Tensor, "... seq_len d_k"], V: Float[Tensor, "... seq_len d_v"]) -> Float[Tensor, "... seq_len d_v"]:
        d_k = Q.shape[-1]
        softmax = Softmax(dim=-1)
        score = (Q @ K.transpose(-2,-1)) / math.sqrt(d_k) 
        if self.mask is None:
            QK_compute = softmax(score)
        else:
            score = score.masked_fill(self.mask==0, -1e4) #1/True = keep, 0/False = block, we can either use -torch.inf or -1e9 or -1e4 or torch.finfo(score.dtype).min
            QK_compute = softmax(score)
        return QK_compute @ V
    

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int,  num_heads, bias = False, device: torch.device | None = None ):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v= d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=bias,device=device)
        self.k_proj = Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = Linear(d_model, d_model, bias=bias, device=device)
        self.o_proj = Linear(d_model, d_model, bias=bias, device=device)

    def forward(self, x: Float[Tensor, " ... seq_len d_in"], token_positions: Int[Tensor, "... seq_len"] | None = None, rope=None,)->Float[Tensor, "... seq_len d_out"]:
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len,seq_len), dtype=torch.bool, device = x.device))
        sdpa = scaled_dot_product_attention(mask)

        # d_k = d_model//num_heads        
        Q:Float[Tensor, "... seq_len d_model"] = self.q_proj(x)
        K:Float[Tensor, "... seq_len d_model"] = self.k_proj(x)
        V:Float[Tensor, "... seq_len d_model"] = self.v_proj(x)


        Q_head:tuple[Float[Tensor, "... seq_len d_k"], ...] = torch.split(Q, int(self.d_k), dim=-1) # torch.split return tuple
        K_head:tuple[Float[Tensor, "... seq_len d_k"], ...] = torch.split(K, int(self.d_k), dim=-1)
        V_head:tuple[Float[Tensor, "... seq_len d_k"], ...] = torch.split(V, int(self.d_v), dim=-1)
        
        heads = []

        for q_h,k_h,v_h in zip(Q_head, K_head, V_head):
            if rope is not None and token_positions is not None:
                q_h:Float[Tensor, "... seq_len d_k"]= rope(q_h, token_positions) 
                k_h:Float[Tensor, "... seq_len d_k"] = rope(k_h, token_positions)
            heads.append(sdpa(q_h, k_h, v_h))

            
        context: Float[Tensor, "... seq_len d_model"] = torch.cat(heads, dim=-1) 
        return self.o_proj(context)    


class transformer_block(nn.Module):
    """
    Args:
    d_model (int): The dimensionality of the Transformer block input.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_ff (int): Dimensionality of the feed-forward inner layer.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    theta (float): RoPE parameter.
    weights (dict[str, Tensor]):
        State dict of our reference implementation.
        The keys of this dictionary are:
        - `attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is (d_model, d_model).
        - `ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `ffn.w3.weight`
            Weight of the third linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
    in_features (Float[Tensor, "batch sequence_length d_model"]):
        Tensor to run your implementation on.
    """

    def __init__(self, 
                d_model:int,
                num_heads:int, 
                d_ff:int, 
                theta: float | None = None,
                max_seq_len: int | None = None,
                bias = False,
                device : torch.device | None = None,
                ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff 
        self.device = device
        self.rope = RoPE(theta,d_model//num_heads, max_seq_len, device= device) if theta is not None and max_seq_len is not None else None
        self.rmsnorm1 = RMSNorm(d_model, device= device)
        self.rmsnorm2 = RMSNorm(d_model, device= device)
        self.MHA_layer = multihead_self_attention(d_model, num_heads, bias= bias, device= device)
        self.FFN = positionwise_feedforward(d_model=d_model,d_ff=d_ff, bias =bias, device= device)

    
    def forward(self, x: Float[Tensor, "... sequence_length d_model"], token_positions = None):
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        x_norm = self.rmsnorm1(x)
        MHA_attn = self.MHA_layer(x_norm, token_positions=token_positions, rope=self.rope)
        h = x + MHA_attn
        h_norm = self.rmsnorm2(h)

        return  self.FFN(h_norm) + h
