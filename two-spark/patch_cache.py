"""Patch the GB10 Inkling image (after gb10/patch_vllm.py) for the two-Spark expert cache (two-spark/gb10_cache.py).

1. ops/gb10_bmm.py: the three expert GEMM kernels read the weight block of slot ``SlotMap[e]`` instead of ``e``
   (``Ones`` and everything else stay per expert). ``bmm(..., slot_map=None)`` uses the identity map, so builds without
   the cache are unchanged.
2. moe.py:
   - INKLING_MOE_TP also splits the MXFP8 layers when the split keeps a multiple of 4 scale columns (TP=2: 36).
   - With INKLING_EXPERT_CACHE=<slots>, the four per-expert weight tensors get <slots> rows, are not read from the
     checkpoint, and are filled from the expert pack; ``forward`` makes the step's experts resident after routing and
     passes the cache's slot map to the GEMMs.
"""
import os, shutil, vllm
d = os.path.join(os.path.dirname(vllm.__file__), "models", "inkling2")
shutil.copy("/opt/inkling2-gb10/gb10_cache.py", os.path.join(d, "ops", "gb10_cache.py"))

p = os.path.join(d, "ops", "gb10_bmm.py"); s = open(p).read()
if "SlotMap" not in s:
    n0 = s.count("    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles,\n")
    s = s.replace("    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles,\n",
                  "    Act, ActScale, RowScale, W, WScale, Ones, Out, Tiles, SlotMap,\n")
    n1 = s.count("    e = tl.load(Tiles + t).to(tl.int64)\n")
    s = s.replace("    e = tl.load(Tiles + t).to(tl.int64)\n",
                  "    e = tl.load(Tiles + t).to(tl.int64)\n    slot = tl.load(SlotMap + e).to(tl.int64)\n")
    for a, b in (("    w_base = W + e * N * (K // 2)\n", "    w_base = W + slot * N * (K // 2)\n"),
                 ("    ws_base = WScale + e * N * (K // 16)\n", "    ws_base = WScale + slot * N * (K // 16)\n"),
                 ("    w_base = W + e * N * K\n", "    w_base = W + slot * N * K\n"),
                 ("    ws_base = WScale + e * N * (K // 32)\n", "    ws_base = WScale + slot * N * (K // 32)\n")):
        assert a in s, a
        s = s.replace(a, b)
    s = s.replace("def bmm(act, act_scale, row_scale, weight, weight_scale, ones, tiles, plan, projection=0, config=None, kernel=None):\n",
                  "_IDENT = {}\n\n\ndef bmm(act, act_scale, row_scale, weight, weight_scale, ones, tiles, plan, projection=0, config=None, kernel=None,\n"
                  "        slot_map=None):\n", 1)
    s = s.replace("    kern[(triton.cdiv(n, bn), m * sub)](\n        *args, ones, out, tiles,\n",
                  "    if slot_map is None:  # no expert cache: slot = expert\n"
                  "        key = (ones.numel(), ones.device)\n"
                  "        slot_map = _IDENT.get(key)\n"
                  "        if slot_map is None:\n"
                  "            slot_map = _IDENT[key] = torch.arange(ones.numel(), dtype=torch.int32, device=ones.device)\n"
                  "    kern[(triton.cdiv(n, bn), m * sub)](\n        *args, ones, out, tiles, slot_map,\n", 1)
    assert n0 == 3 and n1 == 3 and s.count("SlotMap") == 6 and "slot_map, \n" not in s, (n0, n1, s.count("SlotMap"))
    open(p, "w").write(s)

p = os.path.join(d, "moe.py"); s = open(p).read()
if "_CACHE" not in s:
    s = s.replace("_MOE_TP = _os.environ.get('INKLING_MOE_TP') == '1'\n",
                  "_MOE_TP = _os.environ.get('INKLING_MOE_TP') == '1'\n"
                  "from .ops import gb10_cache as _CACHE\n_CACHE_LAYERS = [0]  # layer index of the next Inkling2Experts built\n", 1)
    a = "        self.tp_experts = _MOE_TP and self.ep_size > 1 and quant_format == 'nvfp4'\n"
    assert a in s
    s = s.replace(a, "        # MXFP8 too when the split keeps whole groups of 4 scale columns (TP=2: 2304 / 2 / 32 = 36).\n"
                     "        self.tp_experts = _MOE_TP and self.ep_size > 1 and (quant_format == 'nvfp4' or\n"
                     "            (config.intermediate_size // self.ep_size // 32) % 4 == 0)\n"
                     "        self.layer_index = _CACHE_LAYERS[0]; _CACHE_LAYERS[0] += 1\n"
                     "        self.cache = None\n", 1)
    # per-expert weight tensors: <slots> rows with the cache, and no checkpoint load for them
    a = ("        if per_expert:\n            shape = (self.num_experts, *shape)\n")
    assert a in s
    s = s.replace(a, "        if per_expert:\n"
                     "            rows = self.num_experts\n"
                     "            if _CACHE.enabled() and name in _CACHE.ExpertCache.PIECES:\n"
                     "                rows = min(_CACHE.slots(), self.num_experts)\n"
                     "                loader = lambda param, loaded: None  # filled from the expert pack\n"
                     "            shape = (rows, *shape)\n", 1)
    a = ("        load_kernels(\n            (self.w13_weight, self.w2_weight),\n")
    assert a in s
    s = s.replace(a, "        if _CACHE.enabled():\n"
                     "            assert self.tp_experts and self.ep_size == 2, 'the expert pack is cut for TP=2 with INKLING_MOE_TP=1'\n"
                     "            self.cache = _CACHE.ExpertCache(self.layer_index, self.ep_rank,\n"
                     "                {k: getattr(self, k) for k in _CACHE.ExpertCache.PIECES}, self.num_experts)\n"
                     "            self.cache.fill(_CACHE.initial_order(self.layer_index, self.num_experts))\n" + a, 1)
    a = ("    def _gemm(self, quantized, name: str, tiles, plan, projection: int):\n")
    assert a in s
    s = s.replace("            plan,\n            projection,\n        )\n",
                  "            plan,\n            projection,\n            slot_map=None if self.cache is None else self.cache.slot_map,\n        )\n", 1)
    a = "        src2dst, gammas, rows, tiles = routing\n        gate_up = self._gemm(quantized, \"w13\", tiles, plan, 0)\n"
    assert a in s
    s = s.replace(a, "        src2dst, gammas, rows, tiles = routing\n"
                     "        if self.cache is not None:\n"
                     "            self.cache.ensure(tiles, plan.max_ctas)\n"
                     "        gate_up = self._gemm(quantized, \"w13\", tiles, plan, 0)\n", 1)
    assert s.count("_CACHE") >= 6 and "self.cache.ensure" in s and "slot_map=None if self.cache" in s
    open(p, "w").write(s)
# model.py (the loader accepts pack-filled expert tensors): patches/model.diff
print("patch_cache applied")
