"""gb10_attention.rel_attention vs a dense PyTorch reference: varlen, paged, causal, window, relative bias."""
import math, time
import torch
import gb10_attention as ga

torch.manual_seed(0)
dev = "cuda"
H, KVH, D, PAGE, REL = 48, 8, 128, 128, 1024


def run(reqs, window, dtype=torch.bfloat16):
    """reqs: list of (q_len, kv_len) with kv_len >= q_len (the last q_len keys are the new tokens)."""
    T = sum(q for q, _ in reqs)
    blocks_per = [math.ceil(kv / PAGE) for _, kv in reqs]
    nb = sum(blocks_per) + 3
    kc = torch.randn(nb, PAGE, KVH, D, device=dev, dtype=dtype)
    vc = torch.randn(nb, PAGE, KVH, D, device=dev, dtype=dtype)
    perm = torch.randperm(nb, device=dev)
    bt = torch.zeros(len(reqs), max(blocks_per), dtype=torch.int32, device=dev)
    i = 0
    for r, n in enumerate(blocks_per):
        bt[r, :n] = perm[i:i + n]; i += n
    q = torch.randn(T, H, D, device=dev, dtype=dtype)
    rel = (torch.randn(T, H, REL, device=dev, dtype=torch.float32) * 2).bfloat16().float()
    # vLLM's sheared layout: column pos - d - offset(i), NaN elsewhere
    SW = REL + 256
    sheared = torch.full((T + 128, H, SW), float("nan"), device=dev, dtype=torch.bfloat16)
    t0 = 0
    for ql, kv in reqs:
        for i in range(ql):
            past = kv - ql; pos = past + i
            tile_end = min((i // 128 + 1) * 128, ql)
            off = ((past + tile_end + 127) // 128 - SW // 128) * 128
            d = torch.arange(REL, device=dev); col = pos - d - off
            ok = (col >= 0) & (col < SW) & (d <= pos)
            sheared[t0 + i, :, col[ok]] = rel[t0 + i][:, d[ok]].bfloat16()
        t0 += ql
    cu = torch.tensor([0] + list(torch.tensor([x for x, _ in reqs]).cumsum(0)), dtype=torch.int32, device=dev)
    lens = torch.tensor([kv for _, kv in reqs], dtype=torch.int32, device=dev)
    scale = 1 / math.sqrt(D)
    out = torch.empty(T, H, D, device=dev, dtype=torch.float32)
    lse = torch.empty(H, T, device=dev, dtype=torch.float32)
    ga.rel_attention(q, kc, vc, block_table=bt, cache_seqlens=lens, cu_seqlens_q=cu, max_seqlen_q=max(x for x, _ in reqs),
                     softmax_scale=scale, window_size=(window, 0) if window >= 0 else (-1, -1), sheared=sheared, out=out, lse=lse)
    torch.cuda.synchronize()
    # reference
    ref_o = torch.empty_like(out); ref_l = torch.empty_like(lse)
    for r, (ql, kv) in enumerate(reqs):
        pages = bt[r, :blocks_per[r]].long()
        K = kc[pages].reshape(-1, KVH, D)[:kv].float()
        V = vc[pages].reshape(-1, KVH, D)[:kv].float()
        K = K.repeat_interleave(H // KVH, 1); V = V.repeat_interleave(H // KVH, 1)
        qs = q[cu[r]:cu[r + 1]].float()
        s = torch.einsum("qhd,khd->hqk", qs, K) * scale
        pq = torch.arange(kv - ql, kv, device=dev)[:, None]; pk = torch.arange(kv, device=dev)[None, :]
        dist = pq - pk
        valid = dist >= 0
        if window >= 0:
            valid &= dist <= window
        inrel = valid & (dist < REL)
        b = rel[cu[r]:cu[r + 1]].permute(1, 0, 2).gather(2, dist.clamp(0, REL - 1).expand(H, -1, -1))
        s = s + torch.where(inrel, b, 0.0)
        s = s.masked_fill(~valid, float("-inf"))
        ref_l[:, cu[r]:cu[r + 1]] = torch.logsumexp(s, -1)
        ref_o[cu[r]:cu[r + 1]] = torch.einsum("hqk,khd->qhd", s.softmax(-1), V)
    eo = ((out - ref_o).norm() / ref_o.norm()).item()
    el = (lse - ref_l).abs().max().item()
    t0 = time.perf_counter()
    for _ in range(5):
        ga.rel_attention(q, kc, vc, block_table=bt, cache_seqlens=lens, cu_seqlens_q=cu, max_seqlen_q=max(x for x, _ in reqs),
                         softmax_scale=scale, window_size=(window, 0) if window >= 0 else (-1, -1), sheared=sheared, out=out, lse=lse)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1e3
    print(f"reqs {reqs} window {window}: out rel err {eo:.2e}, lse max abs err {el:.2e}, {ms:.2f} ms")


for window in (-1, 1023):
    run([(1, 1)], window)
    run([(7, 7), (1, 300), (200, 1500)], window)
    run([(1000, 1000)], window)
    run([(300, 2000), (1, 1)], window)
    run([(1, 5000), (1, 20000), (3, 900)], window)
