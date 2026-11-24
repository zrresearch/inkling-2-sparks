"""TP-sharded experts on one GB10: one real NVFP4 MoE layer run whole, then as TP slices (each slice: every expert,
1/TP of the intermediate dimension; w13 rows and w2 K columns), with the slices' bf16 outputs summed as the
all-reduce would. Both are compared with test_moe_layer's no-activation-quantization reference.

    python3 test_tp_layer.py <shard.safetensors> [layer] [tp]
"""
import sys

import torch

sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0", "1"] + sys.argv[3:4]
TP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
sys.argv = sys.argv[:4]
import test_moe_layer as t  # runs its own 1-token check on import  # noqa: E402

I_LOCAL = t.I // TP


def tp_slice(r):
    w13, s13, a13, d13 = t.W["w13"]
    w2, s2, a2, d2 = t.W["w2"]
    rows = slice(r * 2 * I_LOCAL, (r + 1) * 2 * I_LOCAL)
    return {"w13": (w13[:, rows].contiguous(), s13[:, rows].contiguous(), a13, d13),
            "w2": (w2[:, :, r * I_LOCAL // 2:(r + 1) * I_LOCAL // 2].contiguous(),
                   s2[:, :, r * I_LOCAL // 16:(r + 1) * I_LOCAL // 16].contiguous(), a2, d2)}


def ours_with(W, x):
    saved = t.W
    t.W = W
    try:
        return t.ours(x)
    finally:
        t.W = saved


for tokens in (1, 8, 64, 512):
    x = torch.randn(tokens, t.H, device=t.dev)
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * t.norm_w.float()).bfloat16().contiguous()
    ref_bf, _ = t.reference(x, False)
    full = t.ours(x)
    parts = [ours_with(tp_slice(r), x) for r in range(TP)]
    tp = torch.stack([p.float() for p in parts]).sum(0).bfloat16()
    print(f"tokens {tokens:4d}: whole vs ref {t.rel(full, ref_bf):.3e} | TP{TP} vs ref {t.rel(tp, ref_bf):.3e} | "
          f"TP vs whole {t.rel(tp, full):.3e}")
