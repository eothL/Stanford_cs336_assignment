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

                if is_causal:
                    q_idx = i * Bq + torch.arange(0, Bq, device=Q.device)
                    k_idx = j * Bk + torch.arange(0, Bk, device=Q.device)
                    s_j = torch.where(q_idx[:, None] >= k_idx[None, :], s_j, -1e6)

                m_j = torch.maximum(m_i, torch.amax(s_j, dim=-1))
                alpha = torch.exp(m_i - m_j)
                P_tilde = torch.exp(s_j - m_j.unsqueeze(-1))
                l_i = alpha * l_i + torch.sum(P_tilde, dim=-1)
                o_i = alpha.unsqueeze(-1) * o_i + P_tilde @ v_j 
                m_i = m_j
            
            o_i = o_i / l_i.unsqueeze(-1)
            L[:, i*Bq:(i+1)*Bq] = m_i + torch.log(l_i)
            O[:, i*Bq:(i+1)*Bq, :] = o_i

        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    @torch.compile
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        O, L, Q, K, V = ctx.saved_tensors
        _, Nq, d = Q.shape
        scale = d**(-0.5)
        S = Q@K.transpose(-2, -1) * scale
        if ctx.is_causal :
            Nk = K.shape[-2]
            q_idx = torch.arange(Nq, device=Q.device)
            k_idx = torch.arange(Nk, device=Q.device)
            S = torch.where(q_idx[:, None] >= k_idx[None, :], S, -1e6)

        P = torch.exp(S - L.unsqueeze(-1))
        dO = grad_outputs[0]
        dV = P.transpose(-2,-1) @ dO
        dP = dO@V.transpose(-2,-1)
        D_i = (O*dO).sum(dim=-1, keepdim= True)
        dS = P *(dP - D_i)
        dQ = (dS@K) * scale
        dK = (dS.transpose(-2, -1)@Q) * scale
        return (dQ, dK, dV, None, None, None)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
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
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape= (K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
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

        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.where( q_idx[:, None] >= k_idx[None, :], s, -1e6) # same as with additive bias s = s + tl.where( q_idx[:, None] >= k_idx[None, :], 0, -1e6)

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
    
@triton.jit
def d_bwd_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES, 
    D: tl.constexpr,
    Q_TILE_SIZE : tl.constexpr,
):
    output_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1) 

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES,D),
        strides = (stride_oq, stride_od),
        offsets = (output_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1,0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + batch_index * stride_dob, # start at the right batch for dO_ptr 
        shape = (N_QUERIES, D), # matrix size
        strides = (stride_doq, stride_dod), 
        offsets = (output_tile_index * Q_TILE_SIZE,0), # start at the right tile
        block_shape = (Q_TILE_SIZE, D), # block size
        order = (1, 0), # row major matrix
    )
    D_block_ptr = tl.make_block_ptr(
        base = D_ptr + batch_index * stride_db,
        shape = (N_QUERIES,),
        strides = (stride_dq,),
        offsets = (output_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )

    O = tl.load(O_block_ptr, boundary_check = (0, 1), padding_option = "zero")
    dO = tl.load(dO_block_ptr, boundary_check = (0, 1), padding_option = "zero")

    D_i = tl.sum(O * dO, axis=-1) # (Bq, )

    tl.store(D_block_ptr, D_i, boundary_check = (0,))


@triton.jit
def kv_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, D_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    L_ptr, 
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_blk_ptr = tl.make_block_ptr(
        base = K_ptr + batch_index * stride_kb,
        offsets = (key_tile_index * K_TILE_SIZE, 0),
        strides = (stride_kk, stride_kd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    V_blk_ptr = tl.make_block_ptr(
        base = V_ptr + batch_index * stride_vb,
        offsets = (key_tile_index * K_TILE_SIZE, 0),
        strides = (stride_vk, stride_vd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    dK_blk_ptr = tl.make_block_ptr(
        base = dK_ptr + batch_index * stride_dkb,
        offsets = (key_tile_index * K_TILE_SIZE, 0),
        strides = (stride_dkk, stride_dkd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    dV_blk_ptr = tl.make_block_ptr(
        base = dV_ptr + batch_index * stride_dvb,
        offsets = (key_tile_index * K_TILE_SIZE, 0),
        strides = (stride_dvk, stride_dvd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    dO_blk_ptr = tl.make_block_ptr(
        base = dO_ptr + batch_index * stride_dob,
        offsets = (0, 0),
        strides = (stride_doq, stride_dod),
        block_shape = (Q_TILE_SIZE, D),
        shape = (N_QUERIES, D),
        order = (1, 0),
    )
    Q_blk_ptr = tl.make_block_ptr(
        base = Q_ptr + batch_index * stride_qb,
        offsets = (0, 0),
        strides = (stride_qq, stride_qd),
        block_shape = (Q_TILE_SIZE, D),
        shape = (N_QUERIES, D),
        order = (1, 0),
    )
    D_blk_ptr = tl.make_block_ptr(
        base = D_ptr + batch_index * stride_db,
        offsets = (0,),
        strides = (stride_dq,),
        block_shape = (Q_TILE_SIZE,),
        shape = (N_QUERIES,),
        order = (0, ),
    )
    L_blk_ptr = tl.make_block_ptr(
        base = L_ptr + batch_index * stride_lb,
        offsets = (0,),
        strides = (stride_lq,),
        block_shape = (Q_TILE_SIZE,),
        shape = (N_QUERIES,),
        order = (0, ),
    )

    Tq = N_QUERIES // Q_TILE_SIZE
    k_i = tl.load(K_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
    v_i = tl.load(V_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
    dv_i = tl.zeros((K_TILE_SIZE, D), tl.float32)
    dk_i = tl.zeros((K_TILE_SIZE, D), tl.float32)

    """
    for optimization, for causal when (i+1)*bq <= key_tile_index*Bk the entier mask is False (all queries before all keys) -> contribute are zero
    i_start = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE if is_causal else 0
    for i in range(i_start, Tq):
    
    or in a more optimize way to also move the pointer:
    if is_causal:
        i_start = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
    else:
        i_start = 0

    # Advance Q/dO/D/L pointers by i_start * Q_TILE_SIZE so the loop
    # starts at the right tile.
    Q_blk_ptr  = Q_blk_ptr.advance((i_start * Q_TILE_SIZE, 0))
    dO_blk_ptr = dO_blk_ptr.advance((i_start * Q_TILE_SIZE, 0))
    D_blk_ptr  = D_blk_ptr.advance((i_start * Q_TILE_SIZE,))
    L_blk_ptr  = L_blk_ptr.advance((i_start * Q_TILE_SIZE,))

    so instead of Tk x Tq iterations, we have Tk x Tq - Tk(Tk-1)/2 = Tk x tq/2
    """
    for i in range(Tq):
        q_i = tl.load(Q_blk_ptr, boundary_check = (0, 1), padding_option = "zero")      
        do_i = tl.load(dO_blk_ptr, boundary_check = (0, 1), padding_option = "zero")    
        d_i = tl.load(D_blk_ptr, boundary_check = (0,), padding_option = "zero")        
        l_i = tl.load(L_blk_ptr, boundary_check = (0,), padding_option = "zero")        
        s = tl.dot(q_i, tl.trans(k_i)) * scale

        if is_causal:
            # global matrix indices compare to local indices so query doesn't attend to future tile
            q_indx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.where(q_indx[:, None] >= k_indx[None, :], s, -1e6)

        p_i = tl.exp(s - l_i[:, None])

        dv_i = tl.dot(tl.trans(p_i).to(do_i.dtype), do_i, acc= dv_i)
        dP = tl.dot(do_i, tl.trans(v_i))
        dS = p_i *(dP - d_i[:, None])
        dk_i = tl.dot(tl.trans(dS).to(q_i.dtype), q_i, acc= dk_i)

        Q_blk_ptr = Q_blk_ptr.advance((Q_TILE_SIZE, 0))
        dO_blk_ptr = dO_blk_ptr.advance((Q_TILE_SIZE, 0))
        D_blk_ptr = D_blk_ptr.advance((Q_TILE_SIZE,))
        L_blk_ptr = L_blk_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_blk_ptr, (dk_i * scale).to(dK_blk_ptr.type.element_ty), boundary_check = (0, 1))
    tl.store(dV_blk_ptr, dv_i.to(dV_blk_ptr.type.element_ty), boundary_check = (0, 1))

@triton.jit
def q_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, D_ptr,
    dQ_ptr, dO_ptr,
    L_ptr, 
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_blk_ptr = tl.make_block_ptr(
        base = Q_ptr + batch_index * stride_qb,
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        strides = (stride_qq, stride_qd),
        block_shape = (Q_TILE_SIZE, D),
        shape = (N_QUERIES, D),
        order = (1, 0),
    )
    dQ_blk_ptr = tl.make_block_ptr(
        base = dQ_ptr + batch_index * stride_dqb,
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        strides = (stride_dqq, stride_dqd),
        block_shape = (Q_TILE_SIZE, D),
        shape = (N_QUERIES, D),
        order = (1, 0),
    )
    K_blk_ptr = tl.make_block_ptr(
        base = K_ptr + batch_index * stride_kb,
        offsets = (0, 0),
        strides = (stride_kk, stride_kd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    V_blk_ptr = tl.make_block_ptr(
        base = V_ptr + batch_index * stride_vb,
        offsets = (0, 0),
        strides = (stride_vk, stride_vd),
        block_shape = (K_TILE_SIZE, D),
        shape = (N_KEYS, D),
        order = (1, 0),
    )
    dO_blk_ptr = tl.make_block_ptr(
        base = dO_ptr + batch_index * stride_dob,
        offsets = (0, 0),
        strides = (stride_doq, stride_dod),
        block_shape = (Q_TILE_SIZE, D),
        shape = (N_QUERIES, D),
        order = (1, 0),
    )
    D_blk_ptr = tl.make_block_ptr(
        base = D_ptr + batch_index * stride_db,
        offsets = (query_tile_index * Q_TILE_SIZE,),
        strides = (stride_dq,),
        block_shape = (Q_TILE_SIZE,),
        shape = (N_QUERIES,),
        order = (0, ),
    )
    L_blk_ptr = tl.make_block_ptr(
        base = L_ptr + batch_index * stride_lb,
        offsets = (query_tile_index * Q_TILE_SIZE,),
        strides = (stride_lq,),
        block_shape = (Q_TILE_SIZE,),
        shape = (N_QUERIES,),
        order = (0, ),
    )

    Tk = N_KEYS // K_TILE_SIZE
    
    q_i = tl.load(Q_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
    l_i = tl.load(L_blk_ptr, boundary_check = (0,), padding_option = "zero")
    d_i = tl.load(D_blk_ptr, boundary_check = (0,), padding_option = "zero")
    do_i = tl.load(dO_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
    dq_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    """
    for causal early stop, we can stop the K loop earlier
    if is_causal: 
        j_end = (query_tile_index * Q_TILE_SIZE) // K_TILE_SIZE + 1
    else: 
        j_end = Tk

    for i in range(Tk):
    """
    for i in range(Tk):
        k_i = tl.load(K_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
        v_i = tl.load(V_blk_ptr, boundary_check = (0, 1), padding_option = "zero")
        s = tl.dot(q_i, tl.trans(k_i)) * scale

        if is_causal:
            # compare global query tile matrix to local key tile matrix 
            q_indx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.where(q_indx[:, None] >= k_indx[None, :], s, -1e6)

        p_i = tl.exp(s - l_i[:, None])

        dP = tl.dot(do_i, tl.trans(v_i))
        dS = p_i *(dP - d_i[:, None])
        dq_i = tl.dot(dS.to(k_i.dtype), k_i, acc =dq_i)

        K_blk_ptr = K_blk_ptr.advance((K_TILE_SIZE, 0))
        V_blk_ptr = V_blk_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_blk_ptr, (dq_i * scale).to(dQ_blk_ptr.type.element_ty), boundary_check = (0, 1))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False) -> Any:
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = d**(-0.5)

        assert d == K.shape[-1], "Dimension mismatch"

        O = torch.zeros_like(Q)
        L = torch.zeros((B, Nq), device=Q.device, dtype=torch.float32)
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
        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal
        return  O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        O, L, Q, K, V = ctx.saved_tensors
        dO = grad_outputs[0]
        B, Nq, D = Q.shape
        Nk = K.shape[-2]
        No = O.shape[-2]
        scale = D**(-0.5)
        Bq, Bk = ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE

        # init matrix
        D_buf = torch.zeros((B, Nq), device = Q.device, dtype = torch.float32)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dQ = torch.zeros_like(Q)

        # init grid
        grid_o = (triton.cdiv(No, Bq), B)
        grid_k = (triton.cdiv(Nk, Bk), B)
        grid_q = (triton.cdiv(Nq, Bq), B)

        d_bwd_kernel[grid_o](
            O, dO, D_buf,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D_buf.stride(0), D_buf.stride(1),
            N_QUERIES = Nq,
            D = D,
            Q_TILE_SIZE = ctx.Q_TILE_SIZE,
        )
        kv_bwd_kernel[grid_k](
            Q, K, V, D_buf,
            dK, dV, dO,
            L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D_buf.stride(0), D_buf.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES = Nq,
            N_KEYS = Nk,
            scale = scale,
            is_causal = ctx.is_causal,
            Q_TILE_SIZE = ctx.Q_TILE_SIZE,
            K_TILE_SIZE = ctx.K_TILE_SIZE,
            D = D
        )
        q_bwd_kernel[grid_q](
            Q, K, V, D_buf,
            dQ, dO,
            L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D_buf.stride(0), D_buf.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES = Nq,
            N_KEYS = Nk,
            scale = scale,
            is_causal = ctx.is_causal,
            Q_TILE_SIZE = ctx.Q_TILE_SIZE,
            K_TILE_SIZE = ctx.K_TILE_SIZE,
            D = D
        )

        return dQ, dK, dV, None

        
def main():
    return

if __name__=="__main__":
    main()
