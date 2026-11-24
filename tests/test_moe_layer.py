"""Phase 1: one Inkling 2 Small MoE layer on GB10.

vLLM's own Triton routing / SwiGLU-quantize / combine kernels + our Triton NVFP4 expert GEMM (gb10_bmm),
checked against an independent PyTorch fp32 reference of the same W4A4 math, and against a reference
without activation quantization (to show the size of the quantization error itself).
Usage: python3 test_moe_layer.py <shard.safetensors> [layer]
"""

import sys
import time

import torch
from safetensors import safe_open

from vllm.models.inkling2.ops import moe as vmoe
from vllm.models.inkling2.ops.activation import swiglu_quantize
from vllm.models.inkling2.ops.trtllm_bmm import ExpertPlan, _TILES

import gb10_bmm

torch.manual_seed(0)
dev = "cuda"
path, layer = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
P = f"model.llm.layers.{layer}.mlp."
TOPK, ROUTE_SCALE, GATE_CAP, UP_CAP, E, H, I = 8, 8.0, 4.0, 25.0, 384, 4608, 2304

f = safe_open(path, "pt", device=dev)
g = lambda k: f.get_tensor(P + k)
gate_w, gate_b = g("gate.weight").contiguous(), g("gate.bias").contiguous()
norm_w = f.get_tensor(f"model.llm.layers.{layer}.mlp_norm.weight")
W = {n: (g(f"experts.{n}_weight").contiguous(), g(f"experts.{n}_weight_scale").contiguous(),
         g(f"experts.{n}_weight_amax").contiguous(), g(f"experts.{n}_weight_outer_scale_denominator").contiguous())
     for n in ("w13", "w2")}
ones = torch.ones(E, dtype=torch.float32, device=dev)
print("denominators", W["w13"][3].item(), W["w2"][3].item(), "amax range", W["w13"][2].min().item(), W["w13"][2].max().item())

# ---------------- PyTorch reference ----------------
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
MIDS = (E2M1[1:] + E2M1[:-1]) / 2


def e2m1_round(v):
    a = v.abs().clamp(max=6.0)
    idx = (a[..., None] > MIDS).sum(-1)
    tie = (a[..., None] == MIDS)
    tie_any = tie.any(-1)
    j = tie.float().argmax(-1)  # tie between codes j and j+1: pick the even one
    idx = torch.where(tie_any, torch.where(j % 2 == 0, j, j + 1), idx)
    return torch.sign(v) * E2M1[idx]


def e2m1_decode(u8):
    lo, hi = (u8 & 15).long(), (u8 >> 4).long()
    val = lambda c: torch.where(c >= 8, -E2M1[c & 7], E2M1[c & 7])
    return torch.stack([val(lo), val(hi)], -1).reshape(*u8.shape[:-1], u8.shape[-1] * 2)


def deq_weight(n, e):
    w, s, amax, den = W[n]
    v = e2m1_decode(w[e]).reshape(w.shape[1], -1, 16) * s[e].float()[..., None]
    return v.reshape(w.shape[1], -1) * (amax[e] / den)


def quant_rows(x):
    """fouroversix NVFP4 fake-quant of fp32 rows [R, K] -> dequantized fp32 (incl. the amax/1536 outer scale)."""
    amax = x.abs().amax(-1, keepdim=True)
    enc = (1536.0 / amax).clamp(max=3.4028234663852886e38)
    dec, outer = 1.0 / enc, amax / 1536.0
    b = x.reshape(x.shape[0], -1, 16)
    bm = b.abs().amax(-1, keepdim=True)
    best, best_err = None, None
    for div in (6.0, 4.0):
        s = (bm * (enc[..., None] / div)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        inv = (1.0 / (s * dec[..., None])).clamp(max=3.4028234663852886e38)
        r = e2m1_round(b * inv) * s * outer[..., None]
        err = (r - b).abs().sum(-1, keepdim=True)
        if best is None:
            best, best_err = r, err
        else:
            take = err < best_err
            best = torch.where(take, r, best)
    return best.reshape(x.shape)


def softcap(x, c):
    return c * torch.tanh(x / c)


def reference(x, quant_act=True):
    logits = x.float() @ gate_w.float().T
    sel = torch.sigmoid(logits) + gate_b
    ids = sel.topk(TOPK, -1).indices
    p = torch.sigmoid(logits.gather(-1, ids))
    gam = p / p.sum(-1, keepdim=True) * ROUTE_SCALE
    out = torch.zeros(x.shape, dtype=torch.float32, device=dev)
    xq = quant_rows(x.float()) if quant_act else x.float()
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gu = (xq[tok] @ deq_weight("w13", e).T).bfloat16().float()
        gate, up = gu[:, 0::2], gu[:, 1::2]
        y = (softcap(gate, GATE_CAP) * torch.sigmoid(gate) * softcap(up, UP_CAP) * gam[tok, slot][:, None])
        y = y.bfloat16().float()
        yq = quant_rows(y) if quant_act else y
        out.index_add_(0, tok, (yq @ deq_weight("w2", e).T).bfloat16().float())
    return out.bfloat16(), ids


# ---------------- vLLM kernels + ours ----------------
def plan_for(tokens):
    tile = next(t for mx, t in _TILES if tokens <= mx)
    rows = tokens * TOPK
    return ExpertPlan(tile, min(E, rows) + (rows - min(E, rows)) // tile, (0, 0))


def ours(x):
    plan = plan_for(x.shape[0])
    counters = torch.zeros(vmoe.NUM_COUNTERS, dtype=torch.int32, device=dev)
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    bounds = torch.tensor([0, E], dtype=torch.int32, device=dev)
    (src2dst, gammas, rows, tiles), q = vmoe.route(
        x, gate_w, gate_b, counters, counts, TOPK, ROUTE_SCALE, bounds, plan, W["w13"][2], W["w13"][3])
    m = plan.max_ctas
    gu = gb10_bmm.bmm(*q, W["w13"][0], W["w13"][1], ones, tiles, plan)
    act = swiglu_quantize(gu, gammas, GATE_CAP, UP_CAP, (rows, tiles[2 * m + 2:], tiles[:m]), plan.tile,
                          W["w2"][2], W["w2"][3])
    down = gb10_bmm.bmm(*act, W["w2"][0], W["w2"][1], ones, tiles, plan)
    return vmoe.combine(down, src2dst, TOPK, torch.bfloat16)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


for tokens in [int(a) for a in sys.argv[3:]] or (1, 4, 8, 32, 128, 384, 768, 2048, 4096):
    x = torch.randn(tokens, H, device=dev)
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w.float()).bfloat16().contiguous()
    ref_q, ids = reference(x, True)
    ref_bf, _ = reference(x, False)
    try:
        y = ours(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            ours(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref_q.float().flatten(), 0).item()
        print(f"tokens {tokens:4d} tile {plan_for(tokens).tile:3d}: vs W4A4 ref rel {rel(y, ref_q):.2e} cos {cos:.6f} | "
              f"W4A4 ref vs no-act-quant ref rel {rel(ref_q, ref_bf):.2e} | ours vs no-act-quant {rel(y, ref_bf):.2e} | {ms:.2f} ms")
    except Exception as ex:  # report and keep going
        print(f"tokens {tokens:4d}: FAIL {type(ex).__name__}: {str(ex)[:400]}")
