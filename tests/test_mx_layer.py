"""One MXFP8 MoE layer (layers 40-47) on GB10: vLLM's Triton routing / SwiGLU / combine + gb10_bmm's MXFP8 GEMM,
against a PyTorch fp32 reference of the same W8A8 math. Usage: test_mx_layer.py <shard> <layer> [tokens...]"""
import sys, time
import torch
from safetensors import safe_open
from vllm.models.inkling2.ops import moe as vmoe
from vllm.models.inkling2.ops.activation import swiglu_quantize
import gb10_bmm

torch.manual_seed(0)
dev = "cuda"
path, layer = sys.argv[1], int(sys.argv[2])
P = f"model.llm.layers.{layer}.mlp."
TOPK, ROUTE_SCALE, GATE_CAP, UP_CAP, E, H = 8, 8.0, 4.0, 25.0, 384, 4608
f = safe_open(path, "pt", device=dev)
g = lambda k: f.get_tensor(P + k).contiguous()
gate_w, gate_b = g("gate.weight"), g("gate.bias")
norm_w = f.get_tensor(f"model.llm.layers.{layer}.mlp_norm.weight")
W = {n: (g(f"experts.{n}_weight"), g(f"experts.{n}_weight_scale")) for n in ("w13", "w2")}
print({n: (w.dtype, tuple(w.shape), s.dtype, tuple(s.shape)) for n, (w, s) in W.items()})
ones = torch.ones(E, dtype=torch.float32, device=dev)


def deq_weight(n, e):
    w, s = W[n]
    return (w[e].float().reshape(w.shape[1], -1, 32) * torch.exp2(s[e].float() - 127)[..., None]).reshape(w.shape[1], -1)


def quant_rows(x):
    """MXFP8 fake-quant (vLLM's _mxfp8_block): round-up power-of-two scale per 32, e4m3 values."""
    b = x.reshape(x.shape[0], -1, 32)
    amax = b.abs().amax(-1)
    raw = amax / 448.0
    bits = raw.view(torch.int32)
    exp = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exp = torch.where(raw <= 2.0 ** -127, torch.zeros_like(exp), exp.clamp(max=254))
    scale = torch.clamp((exp << 23).view(torch.float32), min=2.0 ** -127)
    inv = torch.where(exp == 0, torch.full_like(scale, 2.0 ** 127), 1.0 / scale)
    v = (b * inv[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (v * scale[..., None]).reshape(x.shape)


def softcap(x, c):
    return c * torch.tanh(x / c)


def reference(x):
    logits = x.float() @ gate_w.float().T
    ids = (torch.sigmoid(logits) + gate_b).topk(TOPK, -1).indices
    p = torch.sigmoid(logits.gather(-1, ids))
    gam = p / p.sum(-1, keepdim=True) * ROUTE_SCALE
    out = torch.zeros(x.shape, dtype=torch.float32, device=dev)
    xq = quant_rows(x.float())
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gu = (xq[tok] @ deq_weight("w13", e).T).bfloat16().float()
        gate, up = gu[:, 0::2], gu[:, 1::2]
        y = (softcap(gate, GATE_CAP) * torch.sigmoid(gate) * softcap(up, UP_CAP) * gam[tok, slot][:, None]).bfloat16().float()
        out.index_add_(0, tok, (quant_rows(y) @ deq_weight("w2", e).T).bfloat16().float())
    return out.bfloat16()


def ours(x):
    plan = gb10_bmm.expert_plan(x.shape[0], TOPK, E, False, ((4608, 4608), (4608, 2304)))
    counters = torch.zeros(vmoe.NUM_COUNTERS, dtype=torch.int32, device=dev)
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    bounds = torch.tensor([0, E], dtype=torch.int32, device=dev)
    (src2dst, gammas, rows, tiles), q = vmoe.route(x, gate_w, gate_b, counters, counts, TOPK, ROUTE_SCALE, bounds, plan, None, None)
    m = plan.max_ctas
    gu = gb10_bmm.bmm(*q, *W["w13"], ones, tiles, plan)
    act = swiglu_quantize(gu, gammas, GATE_CAP, UP_CAP, (rows, tiles[2 * m + 2:], tiles[:m]), plan.tile)
    down = gb10_bmm.bmm(*act, *W["w2"], ones, tiles, plan)
    return vmoe.combine(down, src2dst, TOPK, torch.bfloat16)


for tokens in [int(a) for a in sys.argv[3:]] or (1, 8, 128, 768, 2048, 8192):
    x = torch.randn(tokens, H, device=dev)
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w.float()).bfloat16().contiguous()
    try:
        y = ours(x); r = reference(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            ours(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1e3
        rel = ((y.float() - r.float()).norm() / r.float().norm()).item()
        print(f"tokens {tokens:5d} tile {gb10_bmm.expert_plan(tokens, TOPK, E, False, ((4608, 4608), (4608, 2304))).tile:3d}: rel {rel:.2e} | {ms:.2f} ms", flush=True)
    except Exception as ex:
        print(f"tokens {tokens:5d}: FAIL {type(ex).__name__}: {str(ex)[:300]}", flush=True)
