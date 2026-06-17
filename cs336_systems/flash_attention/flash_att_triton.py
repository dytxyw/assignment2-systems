import triton
import torch
import triton.language as tl
import math

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
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    scaler = scale
    o_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1))
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(T_k):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + stride_kb * batch_index,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, j * K_TILE_SIZE),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + stride_vb * batch_index,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(j * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0)
        )
        K_j = tl.load(K_block_ptr, boundary_check=(1,0))
        V_j = tl.load(V_block_ptr, boundary_check=(0,1))
        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scaler
        if is_causal:
            q_indices = tl.arange(0, Q_TILE_SIZE)[:, None]+ query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE)[None, :] + j * K_TILE_SIZE
            mask = q_indices >= k_indices

            S_ij = tl.where(mask, S_ij, float('-inf'))
        
        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        m_scaler = tl.exp(m_i - m_i_new)
        l_i = l_i * m_scaler + tl.sum(P_ij, 1)
        o_i = o_i * m_scaler[:, None] + tl.dot(P_ij.to(tl.float32), V_j.to(tl.float32))
        m_i = m_i_new

    o_i = o_i * (1.0 / l_i[:, None])
    l_i = tl.log(l_i) + m_i

    tl.store(O_block_ptr, o_i.to(O_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, l_i.to(L_ptr.type.element_ty), boundary_check=(0,))

class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal = False):
        batch_size, N_q, D = q.shape
        _, N_k, _ = k.shape

        
        if D <= 32:
            Q_TILE_SIZE = 128
            K_TILE_SIZE = 128
        elif D <= 64:
            Q_TILE_SIZE = 64
            K_TILE_SIZE = 64
        else:
            Q_TILE_SIZE = 32
            K_TILE_SIZE = 32
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        O_ptr = torch.empty_like(q)
        L_ptr = torch.empty((batch_size, N_q), dtype=torch.float32, device=q.device)

        flash_fwd_kernel[(T_q, batch_size)](
            q, k, v,
            O_ptr, L_ptr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2),
            L_ptr.stride(0), L_ptr.stride(1),
            N_q, N_k,
            D ** (-0.5),
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )
        ctx.save_for_backward(q, k, v, O_ptr, L_ptr)
        ctx.is_causal = is_causal
        return O_ptr
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        device = grad_output.device
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)

        batch_size, N_q, d_model = q.shape
        _, N_k, _ = k.shape
        scale = d_model ** (-0.5)

        @torch.compile
        def flash_backward_impl(Q, K, V, O, L, dO, is_causal):
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)
            O = O.to(torch.float32)
            dO = dO.to(torch.float32)
            scale = Q.shape[-1] ** (-0.5)

            D = torch.sum(O * dO, dim=-1)  # (B, N_q)
            S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, N_q, N_k)
            if is_causal:
                N_q = Q.shape[1]
                N_k = K.shape[1]
                q_indices = torch.arange(N_q, device=Q.device)[:, None]  # (1, N_q)
                k_indices = torch.arange(N_k, device=K.device)[None, :]  # (N_k, 1)
                mask = q_indices >= k_indices   
                S = torch.where(mask, S, torch.tensor(float('-inf'), device=device))

            P = torch.exp(S - L.unsqueeze(-1))    # (B, N_q, N_k)

            dV = torch.matmul(P.transpose(-2, -1), dO)  # (B, N_k, D)
            dP = torch.matmul(dO, V.transpose(-2, -1))  # (B, N_q, N_k)
            dS = P * (dP - D.unsqueeze(-1))  # (B, N_q, N_k)
            dQ = torch.matmul(dS, K) *scale  # (B, N_q, D)
            dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
            dQ = dQ.to(q.dtype)
            dK = dK.to(k.dtype)
            dV = dV.to(v.dtype)
            return dQ, dK, dV
            

    # FIX: 调用内部函数并返回结果
        dQ, dK, dV = flash_backward_impl(q, k, v, O, L, grad_output, ctx.is_causal)
        return dQ, dK, dV, None
    