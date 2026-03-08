import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool


DEFAULT_INIT_STD = 0.02

def _init_trunc_normal(tensor: Tensor, std: float = DEFAULT_INIT_STD) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)

# Architecture
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
        
        # (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features,in_features), **self.factory_kwargs)) 
        self.bias = nn.Parameter(torch.empty((out_features,), **self.factory_kwargs)) if bias is True else None

        _init_trunc_normal(self.weight)
        if self.bias is not None:
            _init_trunc_normal(self.bias) 

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
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **self.factory_kwargs))
        
        _init_trunc_normal(self.weight) # fill the matrix from truncated normal distribution between -3 sigma and 3 sigma
    
    def forward(self, token_ids: Int[Tensor, "..."]) -> torch.Tensor:
        return self.weight[token_ids]

## Normalizer
def rms_normalize(input:torch.Tensor, eps:float=1e-5):
    # prevent overflow when applying square to input convert input to float 32
    in_dtype = input.dtype 
    x_fp32 = input.to(torch.float32) # prevent overflow for low precision
    mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rms = x_fp32 * torch.rsqrt(mean_square+eps) #rsqrt is reverser sqrt 1/sqrt(X)
    return rms.to(in_dtype)

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
        # we can add bias
        nn.init.ones_(self.weights)

    def forward(self, x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        return (self.weights * rms_normalize(input=x, eps=self.eps)).to(x.dtype)
    

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, activation_fcn: str = "swiglu", device= None, dtype = None, bias = False):
        super().__init__()
        self.factory_kwargs = {}
        if device is not None:
            self.factory_kwargs["device"] = device
        if dtype is not None:
            self.factory_kwargs["dtype"] = dtype

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(((8/3 * d_model)//64)*64) # keep a multiple of 64 to make a good use of the hardware
        self.activation_fcn = activation_fcn.lower()
        if self.activation_fcn == "swiglu":
            self.w1_proj = Linear(self.d_model, self.d_ff,  **self.factory_kwargs, bias=bias)
            self.w3_proj = Linear(self.d_model, self.d_ff,  **self.factory_kwargs, bias=bias)
            self.w2_proj = Linear(self.d_ff, self.d_model,  **self.factory_kwargs, bias=bias)
            self._forward_impl = self._forward_swiglu
        elif self.activation_fcn == "relu":
            self.w1_proj = Linear(self.d_model, self.d_ff, **self.factory_kwargs, bias=bias)
            self.w2_proj = Linear(self.d_ff, self.d_model,  **self.factory_kwargs, bias=bias)
            self._forward_impl = self._forward_relu
        elif self.activation_fcn == "sq_relu":
            self.w1_proj = Linear(self.d_model, self.d_ff, **self.factory_kwargs, bias=bias)
            self.w2_proj = Linear(self.d_ff, self.d_model,  **self.factory_kwargs, bias=bias)
            self._forward_impl = self._forward_sq_relu
        elif self.activation_fcn == "ramp_relu":
            self.w1_proj = Linear(self.d_model, self.d_ff, **self.factory_kwargs, bias=bias)
            self.w2_proj = Linear(self.d_ff, self.d_model,  **self.factory_kwargs, bias=bias)
            self.register_buffer(
                "ramp_alpha",
                torch.tensor(0.0, dtype=torch.float32, device=device),
                persistent=False,
            )
            self._forward_impl = self._forward_ramp_relu
        else:
            raise ValueError(f"Unknown activation function: {activation_fcn}, only the following activation function are "
                "supported:['swiglu', 'relu', 'sq_relu', 'ramp_relu'] ")

    def forward(self, x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        return self._forward_impl(x=x)
    
    # ReLU activation function
    def _forward_relu(self, x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        h:torch.Tensor = self.w1_proj(x)
        return self.w2_proj(h.clamp_min(0))

    # ReLU^2 activation function 
    def _forward_sq_relu(self,  x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        h:torch.Tensor = self.w1_proj(x)
        h = h.clamp_min(0)
        return self.w2_proj(h.pow(2))

    # Ramp ReLU activation function with cosine ramping between ReLU and ReLU^2 controlled by self.ramp_alpha
    def _forward_ramp_relu(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        h: torch.Tensor = self.w1_proj(x)
        h_relu = h.clamp_min(0)
        alpha = self.ramp_alpha.to(dtype=h_relu.dtype, device=h_relu.device)
        h_mix = (1.0 - alpha) * h_relu + alpha * h_relu.pow(2)
        return self.w2_proj(h_mix)

    def set_ramp_alpha(self, alpha: float) -> None:
        if not hasattr(self, "ramp_alpha"):
            return
        alpha = float(max(0.0, min(1.0, alpha)))
        self.ramp_alpha.fill_(alpha)
    
    #SwiGLU activation function
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
    
    def _forward_swiglu(self,x:Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
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

        # if x has extra leading dimension (e.g head dim), broadcast positional cos/sin across them
        while cos.ndim < x.ndim : 
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
            
        x_even: Float[Tensor, "... seq_len d_k_half"] = x[..., 0::2]
        x_odd: Float[Tensor, "... seq_len d_k_half"]  = x[..., 1::2]

        out: Float[Tensor, "... seq_len d_k"] = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out # (... seq_len d_k)

#@ Activation function
class Softmax(nn.Module):
    """
    Args:
        d_i (int): a dimension i and apply softmax to the i-th dimension of the input tensor
    For numerical stability, we will substract the largest value in the input tensor as softmax operation is invariant to adding any constant c to all inputs
    """
    
    def __init__(self, dim: int=-1):
        super().__init__()
        self.d_i = dim

    def forward(self, x:Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        exp_x_stable = torch.exp(x - x.amax(dim= self.d_i, keepdim=True))
        return exp_x_stable/exp_x_stable.sum(dim= self.d_i, keepdim=True)
    

## Attention
class scaled_dot_product_attention(nn.Module):
    def __init__(self, max_seq_len: int | None = None, device: torch.device | None = None, mask: Bool[Tensor, " ... queries keys"] | None = None):
        super().__init__()
        self.device = device
        self.softmax = Softmax(dim=-1)
        if mask is None:
            self.register_buffer("causal_mask", 
                                torch.tril(torch.ones((max_seq_len, max_seq_len), dtype= torch.bool, device=device)), 
                                persistent = False)
        else:
            self.causal_mask= mask
            # mask:Float[Tensor, "seq_len seq_len"]| None = None,
        
    def forward(
        self,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_v"],
        seq_len: int | None = None,
        tau: Float[Tensor, ""] | None = None,
    ) -> Float[Tensor, "... seq_len d_v"]:
        d_k = Q.shape[-1]
        mask = self.causal_mask[:seq_len, :seq_len] if seq_len is not None else self.causal_mask
        if tau is not None:
            score = tau * (Q @ K.transpose(-2,-1)) / math.sqrt(d_k)
        else: 
            score = (Q @ K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is None:
            QK_compute = self.softmax(score)
        else:
            score = score.masked_fill(mask==0, torch.finfo(score.dtype).min) #1/True = keep, 0/False = block, we can either use -torch.inf or -1e9 or -1e4 or torch.finfo(score.dtype).min
            QK_compute = self.softmax(score)
        return QK_compute @ V
    

def QK_Norm(Q:torch.Tensor, K:torch.Tensor, eps = 1e-5):
    return (rms_normalize(Q,eps), rms_normalize(K,eps))
    

class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        bias: bool = False,
        device: torch.device | None = None,
        use_qk_norm: bool = False,
        use_value_embeddings: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v= d_model // num_heads
        self.device = device
        self.use_value_embeddings = use_value_embeddings
        self.q_proj = Linear(d_model, d_model, bias=bias,device=device)
        self.k_proj = Linear(d_model, d_model, bias=bias, device=device)
        self.v_proj = Linear(d_model, d_model, bias=bias, device=device)
        self.o_proj = Linear(d_model, d_model, bias=bias, device=device)
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.log_tau = nn.Parameter(torch.zeros(num_heads)) if use_qk_norm is True else None
        if self.use_value_embeddings:
            self.value_mix = nn.Parameter(torch.tensor([1.0, 0.0], device=device))
        # if we don't want a learned tau and fixed one, we can use use register_buffer

        self.sdpa = scaled_dot_product_attention(max_seq_len=max_seq_len, device=device)
        self._forward_impl = self._forward_qk_norm if use_qk_norm is True else self._forward_vanilla
    
    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope=None,
        value_embed: Float[Tensor, "... seq_len d_model"] | None = None,
    ) -> Float[Tensor, "... seq_len d_out"]:
        return self._forward_impl(
            x=x,
            token_positions=token_positions,
            rope=rope,
            value_embed=value_embed,
        )

    def _mix_values(
        self,
        values: Float[Tensor, "... seq_len d_model"],
        value_embed: Float[Tensor, "... seq_len d_model"] | None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        if not self.use_value_embeddings:
            return values
        if value_embed is None:
            return self.value_mix[0] * values
        return self.value_mix[0] * values + self.value_mix[1] * value_embed
    
    def _forward_qk_norm(
        self,
        x: Float[Tensor, " ... seq_len d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope=None,
        value_embed: Float[Tensor, "... seq_len d_model"] | None = None,
    ) -> Float[Tensor, "... seq_len d_out"]:
        seq_len = x.shape[-2]

        # d_k = d_model//num_heads        
        Q:Float[Tensor, "... seq_len d_model"] = self.q_proj(x)
        K:Float[Tensor, "... seq_len d_model"] = self.k_proj(x)
        V:Float[Tensor, "... seq_len d_model"] = self.v_proj(x)
        V = self._mix_values(V, value_embed)

        head_shape = torch.Size((*Q.shape[:-1], self.num_heads, self.d_k)) # ... seq_len H d_k
        # reshaping to split into N head and swap seq_len and num_heads
        Q_head: Float[Tensor, "... num_heads seq_len d_k"] = Q.reshape(head_shape).transpose(-2,-3)
        K_head: Float[Tensor, "... num_heads seq_len d_k"] = K.reshape(head_shape).transpose(-2,-3)
        V_head: Float[Tensor, "... num_heads seq_len d_k"] = V.reshape(head_shape).transpose(-2,-3)

        lead_dim = Q_head.ndim - 3
        tau = torch.exp(self.log_tau).view((*([1] * lead_dim), self.num_heads, 1, 1)) 

        Q_head, K_head = QK_Norm(Q = Q_head, K = K_head)
        if rope is not None and token_positions is not None:
            Q_head:Float[Tensor, "... num_heads seq_len d_k"]= rope(Q_head, token_positions) 
            K_head:Float[Tensor, "... num_heads seq_len d_k"] = rope(K_head, token_positions)

        heads: Float[Tensor, "... num_heads seq_len d_k"] = self.sdpa(Q_head, K_head, V_head, seq_len=seq_len, tau=tau)
        context: Float[Tensor, "... seq_len d_model"] = heads.movedim(-3,-2).reshape(*x.shape[:-1], self.num_heads * self.d_k)
        return self.o_proj(context)    
    
    def _forward_vanilla(
        self,
        x: Float[Tensor, " ... seq_len d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope=None,
        value_embed: Float[Tensor, "... seq_len d_model"] | None = None,
    ) -> Float[Tensor, "... seq_len d_out"]:
        seq_len = x.shape[-2]

        # d_k = d_model//num_heads        
        Q:Float[Tensor, "... seq_len d_model"] = self.q_proj(x)
        K:Float[Tensor, "... seq_len d_model"] = self.k_proj(x)
        V:Float[Tensor, "... seq_len d_model"] = self.v_proj(x)
        V = self._mix_values(V, value_embed)

        head_shape = torch.Size((*Q.shape[:-1], self.num_heads, self.d_k)) # ... seq_len H d_k
        # reshaping to split into N head and swap seq_len and num_heads
        Q_head: Float[Tensor, "... num_heads seq_len d_k"] = Q.reshape(head_shape).transpose(-2,-3)
        K_head: Float[Tensor, "... num_heads seq_len d_k"] = K.reshape(head_shape).transpose(-2,-3)
        V_head: Float[Tensor, "... num_heads seq_len d_k"] = V.reshape(head_shape).transpose(-2,-3)

        if rope is not None and token_positions is not None:
            Q_head:Float[Tensor, "... num_heads seq_len d_k"]= rope(Q_head, token_positions) 
            K_head:Float[Tensor, "... num_heads seq_len d_k"] = rope(K_head, token_positions)

        heads: Float[Tensor, "... num_heads seq_len d_k"] = self.sdpa(Q_head, K_head, V_head, seq_len=seq_len)
        context: Float[Tensor, "... seq_len d_model"] = heads.movedim(-3,-2).reshape(*x.shape[:-1], self.num_heads * self.d_k)
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
                device : torch.device | None = None,
                bias: bool = False,
                remove_rope : bool = False,
                remove_rmsnorm : bool = False,
                use_post_norm : bool = False,
                use_qk_norm: bool = False,
                activation_fcn:str = "swiglu",
                use_x0_mixing: bool = False,
                use_value_embeddings: bool = False,
                ):
        super().__init__()
        self.device = device
        self.use_x0_mixing = use_x0_mixing
        self.rope = None
        if remove_rope is False and theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta,d_model//num_heads, max_seq_len, device= device) 

        Norm = (lambda: nn.Identity()) if remove_rmsnorm else (lambda: RMSNorm(d_model=d_model, device= device))
        self.rmsnorm1 = Norm()
        self.rmsnorm2 = Norm()

        if self.use_x0_mixing:
            self.x0_mix = nn.Parameter(torch.tensor([1.0, 0.0], device=device))

        self.MHA_layer = multihead_self_attention(
            d_model,
            num_heads,
            max_seq_len,
            bias=bias,
            device=device,
            use_qk_norm=use_qk_norm,
            use_value_embeddings=use_value_embeddings,
        )
        self.FFN = positionwise_feedforward(d_model = d_model, d_ff = d_ff, activation_fcn=activation_fcn,bias = bias, device = device)

        self._forward_impl = self._forward_post if use_post_norm else self._forward_pre 

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions=None,
        x0: Float[Tensor, "... sequence_length d_model"] | None = None,
        value_embed: Float[Tensor, "... sequence_length d_model"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        return self._forward_impl(x, token_positions, x0, value_embed)

    def _mix_with_x0(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        x0: Float[Tensor, "... sequence_length d_model"] | None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        if not self.use_x0_mixing:
            return x
        if x0 is None:
            raise ValueError("x0 must be provided when use_x0_mixing=True")
        return self.x0_mix[0] * x + self.x0_mix[1] * x0

    def _forward_pre(self, x, token_positions, x0=None, value_embed=None):
        attn_input = self._mix_with_x0(x, x0)
        attn_out = self.MHA_layer(
            self.rmsnorm1(attn_input),
            token_positions=token_positions,
            rope=self.rope,
            value_embed=value_embed,
        )
        h = x + attn_out
        return h + self.FFN(self.rmsnorm2(h))

    def _forward_post(self, x, token_positions, x0=None, value_embed=None):
        attn_input = self._mix_with_x0(x, x0)
        attn_out = self.MHA_layer(
            attn_input,
            token_positions=token_positions,
            rope=self.rope,
            value_embed=value_embed,
        )
        h_norm = self.rmsnorm1(x + attn_out)
        return self.rmsnorm2(h_norm + self.FFN(h_norm))

    def set_ramp_alpha(self, alpha: float) -> None:
        self.FFN.set_ramp_alpha(alpha)
    

# Model 
class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size:int,
                d_model:int,
                num_layers:int, 
                num_heads:int,
                d_ff:int, 
                context_length:int, 
                rope_theta:float, 
                remove_rope:bool,
                remove_rmsnorm:bool,
                use_post_norm:bool,
                use_qk_norm:bool,
                bias:bool, 
                activation_fcn:str = "swiglu",
                use_x0_mixing: bool = False,
                num_value_embeddings: int = 0,
                value_embedding_pattern: str = "cycle",
                tied_embedding:bool=False,
                device:torch.device | None = None,
                ):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_value_embeddings = num_value_embeddings
        self.value_embedding_pattern = value_embedding_pattern
        self.embedding = Embedding(num_embeddings= vocab_size, embedding_dim=d_model, device=device, dtype=torch.float32)
        self.value_embeddings = nn.ModuleList(
            [Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=torch.float32) for _ in range(num_value_embeddings)]
        )
        self.value_embedding_plan = self._build_value_embedding_plan()
        self.transformer_blocks = nn.ModuleList(
            [transformer_block(
                d_model = d_model,
                num_heads= num_heads,
                d_ff = d_ff,
                theta = rope_theta,
                max_seq_len= context_length,
                device = device,
                bias= bias,
                remove_rope= remove_rope,
                remove_rmsnorm= remove_rmsnorm,
                use_post_norm=use_post_norm,
                use_qk_norm=use_qk_norm,
                activation_fcn=activation_fcn,
                use_x0_mixing=use_x0_mixing,
                use_value_embeddings=num_value_embeddings > 0,
                ) for _ in range(num_layers)])
        
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device= device, bias=bias)
        if tied_embedding is True:
            self.lm_head.weight =self.embedding.weight

        self.head = nn.Sequential(
            RMSNorm(d_model=d_model, device=device),
            self.lm_head
        )

    def _build_value_embedding_plan(self) -> list[int | None]:
        if self.num_value_embeddings <= 0:
            return [None] * self.num_layers

        pattern = self.value_embedding_pattern.lower()
        if pattern == "cycle":
            return [layer_idx % self.num_value_embeddings for layer_idx in range(self.num_layers)]

        if pattern == "first_last":
            if 2 * self.num_value_embeddings > self.num_layers:
                raise ValueError(
                    "first_last value embedding pattern requires "
                    "2 * num_value_embeddings <= num_layers"
                )
            middle_len = self.num_layers - 2 * self.num_value_embeddings
            return (
                list(range(self.num_value_embeddings))
                + [None] * middle_len
                + list(range(self.num_value_embeddings))
            )

        raise ValueError(f"Unsupported value_embedding_pattern: {self.value_embedding_pattern}")
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        value_embed_inputs = [None] * self.num_layers
        if self.num_value_embeddings > 0:
            embedded_values = [value_embedding(x) for value_embedding in self.value_embeddings]
            value_embed_inputs = [
                None if plan_idx is None else embedded_values[plan_idx]
                for plan_idx in self.value_embedding_plan
            ]

        h = x0 = self.embedding(x)
        for block, value_embed in zip(self.transformer_blocks, value_embed_inputs):
            h = block(h, token_positions=token_positions, x0=x0, value_embed=value_embed)
        logits = self.head(h)
        return logits

    def set_ramp_alpha(self, alpha: float) -> None:
        for block in self.transformer_blocks:
            block.set_ramp_alpha(alpha)
