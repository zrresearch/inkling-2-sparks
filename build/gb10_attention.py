"""Paged relative-bias attention for GB10 / sm_12x: a Triton stand-in for Inkling2's tml_fa4 prefill kernel.

Same contract as ``Inkling2FA4RelAttentionKernel.kernel``: causal varlen attention over a paged KV cache
([num_blocks, block_size, kv_heads, D]), optional sliding window ``(left, 0)``, and Inkling's relative bias

    score(q, k) = (q . k) * scale + rel[q, h, d]   for d = pos(q) - pos(k) in [0, rel_extent), else + 0

``rel`` is read straight from vLLM's sheared table ``(T + 128, H, W)`` (W = rel_extent + 256), written by
``attention_prologue``: for query token i of a request, the key at position k sits at column ``k - offset(i)``,
offset(i) = (ceil((past + tile_end(i)) / 128) - W / 128) * 128, tile_end(i) = min((i // 128 + 1) * 128, q_len);
padding is NaN. Reading it in place avoids an unshear pass (and its repeat_interleave, which asserts on the
padded batches of CUDA graphs).

Writes fp32 ``out`` [T, H, D] and the natural-log LSE ``lse`` [H, T].
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _rel_attn_kernel(
    Q, KC, VC, BlockTable, CacheLens, CuQ, Rel, Out, Lse,
    scale_log2, T,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    stride_rt, stride_rh,
    stride_ot, stride_oh,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, REL: tl.constexpr, SHEAR_W: tl.constexpr,
    WINDOW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    req = tl.program_id(2)
    h = tl.program_id(1)
    kvh = h // (H // KVH)
    q0 = tl.load(CuQ + req)
    q_len = tl.load(CuQ + req + 1) - q0
    mb = tl.program_id(0)
    if mb * BM >= q_len:
        return
    kv_len = tl.load(CacheLens + req)
    offs_m = mb * BM + tl.arange(0, BM)
    m_ok = offs_m < q_len
    pos_q = kv_len - q_len + offs_m  # absolute position of each query row
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (q0 + offs_m)[:, None].to(tl.int64) * stride_qt + h * stride_qh + offs_d[None, :], m_ok[:, None], other=0.0)

    hi = tl.minimum(kv_len, kv_len - q_len + (mb + 1) * BM)  # causal: last key any row of this block sees, + 1
    lo = 0
    lo_key = 0  # first key any row of this block may attend to
    if WINDOW >= 0:
        lo_key = tl.maximum(0, kv_len - q_len + mb * BM - WINDOW)
        lo = (lo_key // BN) * BN
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    rel_row = Rel + (q0 + offs_m)[:, None].to(tl.int64) * stride_rt + h * stride_rh
    tile_end = tl.minimum((offs_m // 128 + 1) * 128, q_len)
    past = kv_len - q_len
    shear_off = (((past + tile_end + 127) // 128) - SHEAR_W // 128) * 128
    for start in range(lo, hi, BN):
        offs_n = start + tl.arange(0, BN)
        # Never load keys outside [lo_key, kv_len): vLLM points pages that slid out of the window at its null block,
        # which is never written and can hold NaN, and 0 * NaN = NaN in the p @ v product.
        n_ok = (offs_n < kv_len) & (offs_n >= lo_key)
        page = tl.load(BlockTable + req * stride_bt + offs_n // PAGE, n_ok, other=0).to(tl.int64)
        slot = offs_n % PAGE
        k = tl.load(KC + page[:, None] * stride_kb + slot[:, None] * stride_ks + kvh * stride_kh + offs_d[None, :], n_ok[:, None], other=0.0)
        v = tl.load(VC + page[:, None] * stride_vb + slot[:, None] * stride_vs + kvh * stride_vh + offs_d[None, :], n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        dist = pos_q[:, None] - offs_n[None, :]
        valid = (dist >= 0) & n_ok[None, :] & m_ok[:, None]
        if WINDOW >= 0:
            valid = valid & (dist <= WINDOW)
        col = offs_n[None, :] - shear_off[:, None]
        in_rel = valid & (dist < REL) & (col >= 0) & (col < SHEAR_W)
        bias = tl.load(rel_row + tl.where(in_rel, col, 0), in_rel, other=0.0).to(tl.float32)
        bias = tl.where(bias != bias, 0.0, bias)
        s = tl.where(valid, s + bias * 1.4426950408889634, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(Out + (q0 + offs_m)[:, None].to(tl.int64) * stride_ot + h * stride_oh + offs_d[None, :], out, m_ok[:, None])
    lse = (m_i + tl.log2(l_i)) / 1.4426950408889634
    tl.store(Lse + h * T + q0 + offs_m, lse, m_ok)


def rel_attention(q, key_cache, value_cache, *, block_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                  softmax_scale, window_size, sheared, out, lse, BM=64, BN=64):
    """``sheared``: vLLM's sheared relative logits (T + 128, H, rel_extent + 256)."""
    rel = sheared
    T, H, D = q.shape
    nb, page, kvh, d2 = key_cache.shape
    assert d2 == D and out.shape == (T, H, D) and lse.shape[0] == H
    left = window_size[0]
    assert window_size[1] in (0, -1), window_size
    num_reqs = cu_seqlens_q.numel() - 1
    grid = (triton.cdiv(max_seqlen_q, BM), H, num_reqs)
    _rel_attn_kernel[grid](
        q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel, out, lse,
        softmax_scale * LOG2E, lse.shape[1],
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        rel.stride(0), rel.stride(1),
        out.stride(0), out.stride(1),
        H=H, KVH=kvh, D=D, PAGE=page, REL=rel.shape[2] - 256, SHEAR_W=rel.shape[2], WINDOW=left, BM=BM, BN=BN,
        num_warps=4, num_stages=2,
    )
