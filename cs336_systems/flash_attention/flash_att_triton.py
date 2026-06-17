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
            strides=(stride_vd, stride_vk),
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