"""FP8 (e4m3, one scale per output row) weights for Inkling 2's BF16 attention projections on GB10.

Decode reads every attention weight once per token; at BF16 that is ~2.5 GB per Spark per token at TP=4, the largest
single cost of a decode step. Stored as e4m3 with a per-row fp32 scale, the read halves.

  quantize(w_bf16) -> (w8 e4m3 [N, K], scale fp32 [N]), registered so ``linear`` finds the scale of any row slice
  linear(x, w8 view) -> bf16 x @ deq(w8).T: a Triton kernel for small batches (decode), dequantize + cuBLAS above it

Row slices (``weight[a:b]``, as Inkling2Attention._project takes them) are views into the registered base tensor;
the scale slice is found from the view's offset into that base.
"""
import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
SMALL_M = 16
_BASES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # base data_ptr -> (base w8, scale)


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (e4m3 [N, K], fp32 scale [N]), scale = row amax / 448."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / E4M3_MAX).clamp(min=1e-12)
    w8 = (wf / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    _BASES[w8.data_ptr()] = (w8, scale)
    return w8, scale


def register(w8: torch.Tensor, scale: torch.Tensor) -> None:
    _BASES[w8.data_ptr()] = (w8, scale)


def scale_of(view: torch.Tensor) -> torch.Tensor:
    """The per-row scales of ``view``, a row slice of a registered e4m3 weight."""
    base_ptr = view.untyped_storage().data_ptr()
    w8, scale = _BASES[base_ptr]
    row = (view.data_ptr() - base_ptr) // (w8.stride(0) * w8.element_size())
    return scale[row:row + view.shape[0]]


@triton.jit
def _fp8_gemv_kernel(X, W, S, Y, M, N: tl.constexpr, K: tl.constexpr, stride_w: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_SCALE: tl.constexpr):
    pid = tl.program_id(0)
    rows_n = pid * BN + tl.arange(0, BN)
    n_ok = rows_n < N
    rm = tl.arange(0, BM)
    m_ok = rm < M
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k_ok = (k0 + kk) < K
        x = tl.load(X + rm[:, None] * K + k0 + kk[None, :], m_ok[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(W + rows_n[:, None].to(tl.int64) * stride_w + k0 + kk[None, :], n_ok[:, None] & k_ok[None, :],
                    other=0.0)
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    if HAS_SCALE:
        acc = acc * tl.load(S + rows_n, n_ok, other=0.0)[None, :]
    y = acc.to(tl.bfloat16)
    tl.store(Y + rm[:, None] * N + rows_n[None, :], y, m_ok[:, None] & n_ok[None, :])


def gemv(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor | None, out: torch.Tensor | None = None,
         bn: int = 32, bk: int = 512, warps: int = 4, stages: int = 2) -> torch.Tensor:
    """Small-batch x @ w.T for e4m3 weights with per-row ``scale``, or bf16 weights with ``scale`` None."""
    m, k = x.shape
    n = w8.shape[0]
    assert m <= SMALL_M and x.is_contiguous() and w8.stride(1) == 1
    y = out if out is not None else torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if scale is None and w8.dtype == torch.bfloat16:
        bk = min(bk, 256)  # bf16 tiles are twice the bytes
    _fp8_gemv_kernel[(triton.cdiv(n, bn),)](x, w8, scale if scale is not None else x, y, m, N=n, K=k,
                                           stride_w=w8.stride(0), BM=16, BN=bn, BK=bk, HAS_SCALE=scale is not None,
                                           num_warps=warps, num_stages=stages)
    return y


def linear(x: torch.Tensor, w8: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 x [M, K] @ deq(w8 view).T, written to ``out`` if given."""
    scale = scale_of(w8)
    x = x.contiguous()
    if x.shape[0] <= SMALL_M:
        return gemv(x, w8, scale, out)
    w = (w8.float() * scale[:, None]).to(torch.bfloat16)  # prompts: compute-bound, dequantize once per call
    if out is not None:
        return torch.mm(x, w.t(), out=out)
    return torch.nn.functional.linear(x, w)
