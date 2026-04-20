import torch
from torch import Tensor
from typing import Any, Tuple
from jaxtyping import Int
import triton
import triton.language as tl

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:Tensor, K:Tensor, V:Tensor, is_causal:bool = False, Bq: int= 16, Bk:int=16) -> Tensor :
        # Q, K, V: (batch, N, d) Assume N divisible by Bq/Bk
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = d ** (-0.5)

        O = torch.zeros_like(Q)
        L = torch.zeros(B, Nq, device= Q.device, dtype= Q.dtype)

        Tq, Tk = Nq // Bq, Nk // Bk
        for i in range(Tq):
            q_i = Q[:, i*Bq:(i+1)*Bq, :] # (B, Bq, d)
            m_i = torch.full((B,Bq), float("-inf"), device= Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, Bq, device= Q.device, dtype= Q.dtype)
            o_i = torch.zeros((B, Bq, d), device= Q.device, dtype= Q.dtype)

            for j in range(Tk):
                k_j = K[:, j*Bk:(j+1)*Bk, :]
                v_j = V[:, j*Bk:(j+1)*Bk, :]
                s_j = (q_i @ k_j.transpose(-2, -1)) * scale
                m_j = torch.maximum(m_i, torch.amax(s_j, dim=-1))
                alpha = torch.exp(m_i - m_j)
                P_tilde = torch.exp(s_j - m_j.unsqueeze(-1))
                l_i = alpha * l_i + torch.sum(P_tilde, dim=-1)
                o_i = alpha.unsqueeze(-1) * o_i + P_tilde @ v_j 
                m_i = m_j
            
            o_i = o_i / l_i.unsqueeze(-1)
            L[:, i*Bq:(i+1)*Bq] = m_i + torch.log(l_i)
            O[:, i*Bq:(i+1)*Bq, :] = o_i

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq, 
    N_QUERIES, N_KEYS,
    scale,                              # 1/sq(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl. constexpr,         # Bq
    K_TILE_SIZE: tl. constexpr,         # Bk
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset Each pointer with the corresponding batch index
    # multiplied with the bacth stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,                # base pointer
        shape = (N_QUERIES, D),                         # full tensor shape
        strides = (stride_qq, stride_qd),               # strides
        offsets =(query_tile_index * Q_TILE_SIZE, 0),   # where tile starts
        block_shape= (Q_TILE_SIZE, D),                  # tile shape
        order=(1, 0),                                   # which axis is contiguous
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kq, stride_kd),
        offsets = (0, 0),
        block_shape= (K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vq, stride_vd),
        offsets =(0, 0),
        block_shape= (K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets =(query_tile_index * Q_TILE_SIZE, 0),
        block_shape= (Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets =(query_tile_index * Q_TILE_SIZE,),
        block_shape= (Q_TILE_SIZE,),
        order=(0,),
    )


    Tk = N_KEYS // K_TILE_SIZE

    # init variable
    q_i = tl.load(Q_block_ptr, boundary_check= (0, 1), padding_option = "zero")
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32) 
    l_i = tl.zeros((Q_TILE_SIZE,), tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    for i in range(Tk):
        k_i = tl.load(K_block_ptr, boundary_check = (0,1), padding_option = "zero")
        v_i = tl.load(V_block_ptr, boundary_check = (0,1), padding_option = "zero")
        s = tl.dot(q_i, tl.trans(k_i)) * scale # (Bq, Bk)
        m_j = tl.maximum(m_i, tl.max(s, axis= -1))
        alpha = tl.exp(m_i - m_j)
        p_tilde = tl.exp(s - m_j[:, None])
        l_i = alpha * l_i +  tl.sum(p_tilde, axis = -1)
        o_i = alpha[:, None] * o_i 
        o_i = tl.dot(p_tilde.to(v_i.dtype), v_i, acc=o_i)
        m_i = m_j


        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]
    tl.store(O_block_ptr, o_i, boundary_check= (0,1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check= (0,))
        

class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False) -> Any:
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = d**(-0.5)

        assert d == K.shape[-1], "Dimension mismatch"

        O = torch.zeros_like(Q)
        L = torch.zeros((B, Nq), device=Q.device, dtype=torch.float32)
        ctx.D_TILE_SIZE = triton.next_power_of_2(d) // 16
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        
        grid = (triton.cdiv(Nq, ctx.Q_TILE_SIZE), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return  O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

        
def main():
    return

if __name__=="__main__":
    main()