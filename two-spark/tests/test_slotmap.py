"""Slot-map kernels vs the original: bit-identical output (1) with no slot map and (2) with experts stored in shuffled
slots plus the matching slot map, for NVFP4 and MXFP8 expert weights. Random weights and activations (equality only
needs both paths to see the same bytes); real routing from vLLM's route kernel. Runs in the tp2cache image with the
original gb10_bmm.py mounted at /data/orig/gb10_bmm.py."""
import sys, importlib.util, torch
sys.path.insert(0, "/data/orig")
spec = importlib.util.spec_from_file_location("orig_bmm", "/data/orig/gb10_bmm.py"); orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
from vllm.models.inkling2.ops import gb10_bmm as new
from vllm.models.inkling2.ops import moe as vmoe
from vllm.models.inkling2.ops.trtllm_bmm import ExpertPlan, _TILES
torch.manual_seed(0); dev = "cuda"
E, H, TOPK = 384, 4608, 8
def plan_for(tokens):
    tile = next(t for mx, t in _TILES if tokens <= mx); rows = tokens * TOPK
    return ExpertPlan(tile, min(E, rows) + (rows - min(E, rows)) // tile, (0, 0))
def rand_e4m3(*shape):  # finite e4m3 bytes in a sane range
    return torch.randint(0x18, 0x48, shape, dtype=torch.uint8, device=dev)
gate_w = (torch.randn(E, H, device=dev) * 0.02).bfloat16(); gate_b = torch.zeros(E, device=dev)
amax = torch.ones(E, device=dev); den = torch.ones(1, device=dev)
ok = True
for tokens in (1, 8, 64, 512):
    plan = plan_for(tokens); m = plan.max_ctas
    x = torch.randn(tokens, H, device=dev).bfloat16()
    counters = torch.zeros(vmoe.NUM_COUNTERS, dtype=torch.int32, device=dev); counts = torch.zeros(E, dtype=torch.int32, device=dev)
    bounds = torch.tensor([0, E], dtype=torch.int32, device=dev)
    (src2dst, gammas, rows, tiles), q = vmoe.route(x, gate_w, gate_b, counters, counts, TOPK, 8.0, bounds, plan, amax, den)
    ones = torch.rand(E, device=dev) + 0.5
    perm = torch.randperm(E, device=dev)                     # slot s holds expert perm[s]
    slot_map = torch.empty(E, dtype=torch.int32, device=dev); slot_map[perm] = torch.arange(E, dtype=torch.int32, device=dev)
    for fmt, N, K in (("nvfp4", 2304, 4608), ("nvfp4", 4608, 1152), ("mxfp8", 2304, 4608), ("mxfp8", 4608, 1152)):
        if fmt == "nvfp4":
            W = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev); Ws = rand_e4m3(E, N, K // 16).view(torch.float8_e4m3fn)
            act, act_s, rs = (q[0], q[1], q[2]) if K == H else (torch.randint(0, 256, (plan.rows, K // 2), dtype=torch.uint8, device=dev),
                              rand_e4m3(plan.rows, K // 16).view(torch.float8_e4m3fn), torch.rand(plan.rows, device=dev))
        else:
            W = rand_e4m3(E, N, K).view(torch.float8_e4m3fn); Ws = torch.randint(120, 130, (E, N, K // 32), dtype=torch.uint8, device=dev)
            act = rand_e4m3(plan.rows, K).view(torch.float8_e4m3fn); act_s = torch.randint(120, 130, (plan.rows, K // 32), dtype=torch.uint8, device=dev); rs = None
        a = orig.bmm(act, act_s, rs, W, Ws, ones, tiles, plan)
        b = new.bmm(act, act_s, rs, W, Ws, ones, tiles, plan)
        c = new.bmm(act, act_s, rs, W[perm].contiguous(), Ws[perm].contiguous(), ones, tiles, plan, slot_map=slot_map)
        # only rows inside a tile are written (the output buffer is torch.empty): row tile t covers [t*TILE, Tiles[MAX_CTAS+t])
        n_t = int(tiles[2 * m].item()); tc = tiles.cpu(); valid = torch.zeros(a.shape[0], dtype=torch.bool)
        for t in range(n_t):
            valid[t * plan.tile: int(tc[m + t])] = True
        valid = valid.to(dev); used = a[valid]
        same_b, same_c = torch.equal(a[valid], b[valid]), torch.equal(a[valid], c[valid])
        ok &= same_b and same_c
        print(f"tokens {tokens:4d} {fmt} N={N} K={K}: no map identical={same_b}, shuffled slots identical={same_c}, "
              f"tiles={n_t}, nonzero outputs={int((used != 0).sum())}", flush=True)
print("ALL IDENTICAL" if ok else "MISMATCH")
