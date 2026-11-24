"""NVFP4 expert GEMM for GB10 / sm_12x: a Triton stand-in for Inkling2's TRT-LLM-gen ``bmm``.

Same contract as ``vllm.models.inkling2.ops.trtllm_bmm.bmm`` for NVFP4, except the weights stay in the
checkpoint's natural layout (rows unshuffled, row-major block scales):

  out[r, n] = row_scale[r] * ones[e] * sum_k deq(act)[r, k] * deq(W[e])[n, k]

with deq(v) = e2m1(v) * e4m3(block scale of 16). Activation scales use the swizzled layout of the plan's
tile (R8c4 below 128 rows, R128c4 at 128+), as written by ``route`` / ``swiglu_quantize``. Rows are
tile-padded: tile t holds rows [t * TILE, tiles[m + t]) of expert tiles[t]; tiles[2m] is the tile count.

fp4 x e4m3 products are exact in bf16, so dequantizing into bf16 and accumulating in fp32 gives the
NVFP4 MMA's result up to summation order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(n):
    """Values of 4-bit e2m1 codes (int32 0..15), with integer ops only: |v| * 4 = 2m (e = 0) or (2 + m) << e."""
    e = (n >> 1) & 3
    m = n & 1
    q = tl.where(e == 0, 2 * m, (2 + m) << e)
    v = q.to(tl.float32) * 0.25
    return tl.where((n & 8) != 0, -v, v)


@triton.jit
def _scale_offsets(row, cols, COLS: tl.constexpr, TILE: tl.constexpr):
    if TILE >= 128:
        tile = (row // 128) * (COLS // 4) + cols // 4
        return tile * 512 + (row % 32) * 16 + (row // 32 % 4) * 4 + cols % 4
    if TILE > 0:
        tile = (row // 8) * (COLS // 4) + cols // 4
        return tile * 32 + (row % 8) * 4 + cols % 4
    return row * COLS + cols


@triton.jit
def _unpack(bytes_):
    """[R, C] uint8 -> [R, 2C] fp32, low nibble first."""
    b = bytes_.to(tl.int32)
    lo = _e2m1(b & 15)
    hi = _e2m1(b >> 4)
    return tl.reshape(tl.join(lo, hi), (bytes_.shape[0], bytes_.shape[1] * 2))


@triton.jit
def _bmm_kernel(
    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles,
    N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr, MAX_CTAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SUB: tl.constexpr,
):
    # Program (n block, tile * SUB + row block): tiles wider than BM are split into SUB row blocks.
    t = tl.program_id(1) // SUB
    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    e = tl.load(Tiles + t).to(tl.int64)
    end = tl.load(Tiles + MAX_CTAS + t)
    rows = t * TILE + (tl.program_id(1) % SUB) * BM + tl.arange(0, BM)
    row_ok = rows < end
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_ok = cols < N
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    w_base = W + e * N * (K // 2)
    ws_base = WScale + e * N * (K // 16)
    for k0 in range(0, K, BK):
        a = tl.load(Act + rows[:, None].to(tl.int64) * (K // 2) + k0 // 2 + kb[None, :], row_ok[:, None], other=0)
        a_s = tl.load(ActScale + _scale_offsets(rows[:, None], k0 // 16 + ks[None, :], K // 16, TILE),
                      row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        a = tl.reshape(tl.reshape(_unpack(a), (BM, BK // 16, 16)) * a_s[:, :, None], (BM, BK))
        w = tl.load(w_base + cols[:, None].to(tl.int64) * (K // 2) + k0 // 2 + kb[None, :], col_ok[:, None], other=0)
        w_s = tl.load(ws_base + cols[:, None].to(tl.int64) * (K // 16) + k0 // 16 + ks[None, :],
                      col_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        w = tl.reshape(tl.reshape(_unpack(w), (BN, BK // 16, 16)) * w_s[:, :, None], (BN, BK))
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)
    rs = tl.load(RowScale + rows, row_ok, other=0.0) * tl.load(Ones + e)
    out = (acc * rs[:, None]).to(tl.bfloat16)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], out, row_ok[:, None] & col_ok[None, :])


@triton.jit
def _bmm_ds_kernel(
    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles,
    N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr, MAX_CTAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SUB: tl.constexpr,
):
    """As ``_bmm_kernel``, with Triton's block-scaled dot (NVFP4: e2m1 values, e4m3 scales per 16)."""
    t = tl.program_id(1) // SUB
    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    e = tl.load(Tiles + t).to(tl.int64)
    end = tl.load(Tiles + MAX_CTAS + t)
    rows = t * TILE + (tl.program_id(1) % SUB) * BM + tl.arange(0, BM)
    row_ok = rows < end
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_ok = cols < N
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    w_base = W + e * N * (K // 2)
    ws_base = WScale + e * N * (K // 16)
    for k0 in range(0, K, BK):
        a = tl.load(Act + rows[:, None].to(tl.int64) * (K // 2) + k0 // 2 + kb[None, :], row_ok[:, None], other=0)
        a_s = tl.load(ActScale + _scale_offsets(rows[:, None], k0 // 16 + ks[None, :], K // 16, TILE),
                      row_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        w = tl.load(w_base + cols[:, None].to(tl.int64) * (K // 2) + k0 // 2 + kb[None, :], col_ok[:, None], other=0)
        w_s = tl.load(ws_base + cols[:, None].to(tl.int64) * (K // 16) + k0 // 16 + ks[None, :],
                      col_ok[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, a_s, "e2m1", tl.trans(w), w_s, "e2m1", acc=acc, rhs_k_pack=True)
    rs = tl.load(RowScale + rows, row_ok, other=0.0) * tl.load(Ones + e)
    out = (acc * rs[:, None]).to(tl.bfloat16)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], out, row_ok[:, None] & col_ok[None, :])



@triton.jit
def _bmm_mx_kernel(
    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles,
    N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr, MAX_CTAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SUB: tl.constexpr,
):
    """MXFP8 experts (e4m3 values, ue8m0 scales per 32 along K): no per-row scale, ``Ones`` per expert."""
    t = tl.program_id(1) // SUB
    if t >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    e = tl.load(Tiles + t).to(tl.int64)
    end = tl.load(Tiles + MAX_CTAS + t)
    rows = t * TILE + (tl.program_id(1) % SUB) * BM + tl.arange(0, BM)
    row_ok = rows < end
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_ok = cols < N
    kk = tl.arange(0, BK)
    ks = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    w_base = W + e * N * K
    ws_base = WScale + e * N * (K // 32)
    for k0 in range(0, K, BK):
        a = tl.load(Act + rows[:, None].to(tl.int64) * K + k0 + kk[None, :], row_ok[:, None], other=0)
        a_s = tl.load(ActScale + _scale_offsets(rows[:, None], k0 // 32 + ks[None, :], K // 32, TILE),
                      row_ok[:, None], other=127)
        w = tl.load(w_base + cols[:, None].to(tl.int64) * K + k0 + kk[None, :], col_ok[:, None], other=0)
        w_s = tl.load(ws_base + cols[:, None].to(tl.int64) * (K // 32) + k0 // 32 + ks[None, :], col_ok[:, None], other=127)
        acc = tl.dot_scaled(a.to(tl.float8e4nv, bitcast=True), a_s, "e4m3",
                            tl.trans(w.to(tl.float8e4nv, bitcast=True)), w_s, "e4m3", acc=acc)
    out = (acc * tl.load(Ones + e)).to(tl.bfloat16)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], out, row_ok[:, None] & col_ok[None, :])


def bmm(act, act_scale, row_scale, weight, weight_scale, ones, tiles, plan, projection=0, config=None, kernel=None):
    """bf16 [plan.rows, N]. NVFP4: ``weight`` uint8 [E, N, K/2], ``weight_scale`` e4m3 [E, N, K/16].
    MXFP8: ``weight`` e4m3 [E, N, K], ``weight_scale`` ue8m0 uint8 [E, N, K/32]. Natural layout (unshuffled)."""
    nvfp4 = weight.dtype == torch.uint8
    n, k = weight.shape[1], weight.shape[2] * (2 if nvfp4 else 1)
    m = plan.max_ctas
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    bm, bn, bk, warps, stages = config or (CONFIGS if nvfp4 else MX_CONFIGS)[plan.tile]
    while k % bk:  # the K loop has no mask: TP-sharded experts give w2 K = 2304 / TP (576 at TP=4)
        bk //= 2
    sub = max(1, plan.tile // bm)
    if nvfp4:
        kern = kernel or (_bmm_ds_kernel if plan.tile >= DS_MIN_TILE else _bmm_kernel)
        args = (act, act_scale.view(torch.uint8), row_scale, weight, weight_scale.view(torch.uint8))
    else:
        kern = kernel or _bmm_mx_kernel
        args = (act.view(torch.uint8), act_scale.view(torch.uint8), ones, weight.view(torch.uint8), weight_scale.view(torch.uint8))
    kern[(triton.cdiv(n, bn), m * sub)](
        *args, ones, out, tiles,
        N=n, K=k, TILE=plan.tile, MAX_CTAS=m, BM=bm, BN=bn, BK=bk, SUB=sub, num_warps=warps, num_stages=stages)
    return out


DS_MIN_TILE = 0  # the block-scaled dot wins at every tile on GB10 (bench_ds.py)
# token tile -> (BM, BN, BK, warps, stages)
CONFIGS = {8: (16, 256, 256, 4, 3), 16: (16, 256, 256, 4, 3), 32: (32, 128, 256, 8, 2),
           64: (16, 128, 256, 8, 2), 128: (64, 64, 256, 8, 2), 256: (64, 64, 256, 8, 2)}
MX_CONFIGS = {8: (16, 128, 128, 4, 3), 16: (16, 128, 128, 4, 3), 32: (32, 128, 128, 4, 2),
              64: (32, 128, 128, 4, 2), 128: (64, 64, 128, 4, 2), 256: (64, 64, 128, 4, 2)}


# ---- vLLM integration: drop-ins for trtllm_bmm.expert_plan / load_kernels ----
import functools

_TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))


@functools.cache
def expert_plan(tokens, topk, num_experts, nvfp4, shapes, tile=None):
    from vllm.models.inkling2.ops.trtllm_bmm import ExpertPlan
    if tile is None:
        tile = next(t for max_tokens, t in _TILES if tokens <= max_tokens)
    rows = tokens * topk
    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    return ExpertPlan(tile, max_ctas, (0, 0))


def load_kernels(weights, weight_scales, topk):
    """Compiles and loads every tile's kernel by running it on no tiles (no loading during CUDA graph capture)."""
    num_experts = weights[0].shape[0]
    nvfp4 = weights[0].dtype == torch.uint8
    shapes = tuple((w.shape[1], w.shape[2] * (2 if nvfp4 else 1)) for w in weights)
    device = weights[0].device
    ones = torch.ones(num_experts, dtype=torch.float32, device=device)
    for _, tile in _TILES:
        plan = expert_plan(1, topk, num_experts, nvfp4, shapes, tile)
        tiles = torch.zeros(2 * plan.max_ctas + 2, dtype=torch.int32, device=device)
        for weight, scale in zip(weights, weight_scales):
            k = weight.shape[2]
            act = torch.empty((plan.rows, k), dtype=weight.dtype, device=device)
            act_scale = torch.empty((plan.rows, scale.shape[2]), dtype=scale.dtype, device=device)
            bmm(act, act_scale, ones.new_empty(plan.rows) if nvfp4 else None, weight, scale, ones, tiles, plan)


@functools.cache
def use_gb10() -> bool:
    """sm_12x (GB10, RTX PRO 6000 Blackwell), read through NVML so CUDA is not initialized at import."""
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    return cap is not None and cap.major == 12
