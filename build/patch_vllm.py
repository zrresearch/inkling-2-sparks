"""Patch the installed vLLM's Inkling 2 MoE to use gb10_bmm on sm_12x (GB10, RTX PRO 6000). Idempotent."""
import os, shutil, vllm
d = os.path.join(os.path.dirname(vllm.__file__), "models", "inkling2")
for f in ("gb10_bmm.py", "gb10_attention.py", "gb10_fp8.py", "expert_stats.py"):
    shutil.copy(os.path.join(os.path.dirname(__file__), f), os.path.join(d, "ops", f))
p = os.path.join(d, "moe.py"); s = open(p).read()
if "gb10_bmm" not in s:
    s = s.replace("from .ops.trtllm_bmm import bmm, expert_plan, load_kernels\n",
                  "from .ops import gb10_bmm\n"
                  "if gb10_bmm.use_gb10():\n"
                  "    from .ops.gb10_bmm import bmm, expert_plan, load_kernels\n"
                  "else:\n"
                  "    from .ops.trtllm_bmm import bmm, expert_plan, load_kernels\n"
                  "_NATURAL_LAYOUT = gb10_bmm.use_gb10()\n")
    s = s.replace("        param.data.copy_(shuffle_rows(param.data))\n",
                  "        if not _NATURAL_LAYOUT:\n            param.data.copy_(shuffle_rows(param.data))\n")
    s = s.replace("        param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))\n",
                  "        if not _NATURAL_LAYOUT:\n            param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))\n")
    assert s.count("_NATURAL_LAYOUT") == 3
    open(p, "w").write(s)
# linear.py: GB10 has 99 KB of shared memory per SM (B200: 228 KB). Skip the 128x128 TMA path and cap pipeline stages.
p = os.path.join(d, "ops", "linear.py"); s = open(p).read()
if "_SM12X" not in s:
    s = s.replace("TMA_MIN_TOKENS = 2048\n",
                  "from .gb10_bmm import use_gb10 as _use_gb10\n_SM12X = _use_gb10()\n"
                  "TMA_MIN_TOKENS = 1 << 62 if _SM12X else 2048\n")
    # FlashInfer's CuTe DSL low-M bf16 GEMM refuses sm_121: cuBLAS for every batch size there.
    s = s.replace("LOW_M_MAX_TOKENS = 32\n", "LOW_M_MAX_TOKENS = 32\n", 1)
    s = s.replace("    if x.shape[0] > LOW_M_MAX_TOKENS or is_flashinfer_autotuning():\n",
                  "    if x.shape[0] > LOW_M_MAX_TOKENS or is_flashinfer_autotuning() or _SM12X:\n", 1)
    s = s.replace("    num_stages = 4 if not split else 3 if block_n <= 64 else 2\n",
                  "    num_stages = 4 if not split else 3 if block_n <= 64 else 2\n"
                  "    if _SM12X:\n        num_stages = min(num_stages, 2)\n")
    assert s.count("_SM12X") == 4, s.count("_SM12X")
    open(p, "w").write(s)
# ops/attention.py: prefill (and, below, all) attention through gb10_attention; no tml_fa4 warm-up compiles.
p = os.path.join(d, "ops", "attention.py"); s = open(p).read()
if "_SM12X" not in s:
    s = s.replace("from vllm.platforms import current_platform\n",
                  "from vllm.platforms import current_platform\n"
                  "from .gb10_bmm import use_gb10 as _use_gb10\n_SM12X = _use_gb10()\n", 1)
    s = s.replace('''        """Write ``out`` and the ``(num_heads, num_tokens)`` LSE ``lse``."""\n''',
                  '''        """Write ``out`` and the ``(num_heads, num_tokens)`` LSE ``lse``."""\n'''
                  "        if _SM12X:\n"
                  "            from .gb10_attention import rel_attention\n"
                  "            rel_attention(q, key_cache, value_cache, block_table=block_table, cache_seqlens=cache_seqlens,\n"
                  "                          cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, softmax_scale=softmax_scale,\n"
                  "                          window_size=window_size, sheared=sheared_bias, out=out, lse=lse)\n"
                  "            return\n", 1)
    s = s.replace("        from torch._subclasses.fake_tensor import FakeTensorMode\n\n        key = compile_key\n",
                  "        from torch._subclasses.fake_tensor import FakeTensorMode\n\n        if _SM12X:\n            return\n        key = compile_key\n", 1)
    assert s.count("_SM12X") == 3, s.count("_SM12X")
    open(p, "w").write(s)
# attention.py: no Blackwell-datacenter decode kernel (Gluon / tensor memory); every token takes the path above.
p = os.path.join(d, "attention.py"); s = open(p).read()
if "_SM12X" not in s:
    s = s.replace('''        leading decode/verify requests."""\n''',
                  '''        leading decode/verify requests."""\n'''
                  "        from .ops.gb10_bmm import use_gb10\n"
                  "        if use_gb10():  # _SM12X\n            return 0, 0, 0\n", 1)
    assert s.count("_SM12X") == 1
    open(p, "w").write(s)
# moe.py: INKLING_MOE_TP=1 shards every expert over the TP group (each rank: all experts, 1/TP of the intermediate
# dimension) instead of whole experts (EP). Every rank then does the same work in every layer; with EP the token's
# experts land unevenly and the all-reduce waits for the busiest rank. Gate/up rows are interleaved pairs, so a rank's
# w13 slice is one contiguous block of rows and its w2 slice one block of K columns.
p = os.path.join(d, "moe.py"); s = open(p).read()
if "_MOE_TP" not in s:
    s = s.replace("_NATURAL_LAYOUT = gb10_bmm.use_gb10()\n",
                  "_NATURAL_LAYOUT = gb10_bmm.use_gb10()\n"
                  "import os as _os\n_MOE_TP = _os.environ.get('INKLING_MOE_TP') == '1'\n", 1)
    s = s.replace("        assert config.n_routed_experts % self.ep_size == 0\n"
                  "        self.num_experts = config.n_routed_experts // self.ep_size\n",
                  "        # NVFP4 layers only: MXFP8's swizzled activation scales group scale columns in fours, and\n"
                  "        # 2304 / TP / 32 = 18 at TP=4 does not divide (colliding writes, wrong and nondeterministic).\n"
                  "        self.tp_experts = _MOE_TP and self.ep_size > 1 and quant_format == 'nvfp4'\n"
                  "        if self.tp_experts:\n"
                  "            assert config.intermediate_size % (64 * self.ep_size) == 0\n"
                  "            self.num_experts = config.n_routed_experts\n"
                  "        else:\n"
                  "            assert config.n_routed_experts % self.ep_size == 0\n"
                  "            self.num_experts = config.n_routed_experts // self.ep_size\n", 1)
    s = s.replace("        hidden, inter = config.hidden_size, config.intermediate_size\n",
                  "        hidden, inter = config.hidden_size, config.intermediate_size\n"
                  "        if self.tp_experts:\n            inter //= self.ep_size\n", 1)
    s = s.replace("        start = self.ep_rank * self.num_experts\n        self.register_buffer(\n",
                  "        start = 0 if self.tp_experts else self.ep_rank * self.num_experts\n        self.register_buffer(\n", 1)
    s = s.replace("    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:\n",
                  "    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:\n"
                  "        if self.tp_experts:\n"
                  "            r = self.ep_rank\n"
                  "            if param is self.w13_weight or param is self.w13_weight_scale:\n"
                  "                n = param.shape[1]\n                loaded = loaded[:, r * n : (r + 1) * n]\n"
                  "            elif param is self.w2_weight or param is self.w2_weight_scale:\n"
                  "                c = param.shape[2]\n                loaded = loaded[:, :, r * c : (r + 1) * c]\n"
                  "            self._load_full(param, loaded.contiguous())\n"
                  "            return\n", 1)
    assert s.count("_MOE_TP") == 3 and s.count("self.tp_experts") == 5, (s.count("_MOE_TP"), s.count("self.tp_experts"))
    open(p, "w").write(s)
# ops/activation.py: MXFP8 SwiGLU at <= 512 rows runs a program per 256 columns and asserts 256 | n. With TP-sharded
# experts n = 2304 / TP (576 at TP=4): use the largest power-of-two chunk up to 256 dividing n (whole 32-col blocks).
p = os.path.join(d, "ops", "activation.py"); s = open(p).read()
if "_TP_CHUNK" not in s:
    s = s.replace("        assert n % 256 == 0\n        head, tail, num_warps = 256, 0, 2\n",
                  "        head = 256  # _TP_CHUNK\n        while n % head:\n            head //= 2\n"
                  "        assert head >= 32\n        tail, num_warps = 0, 2\n", 1)
    assert s.count("_TP_CHUNK") == 1
    open(p, "w").write(s)
# ops/linear.py + attention.py: the attention projections' decode GEMVs.
#   INKLING_ATTN_GEMV=1: batches of <= 16 rows on gb10_fp8's Triton kernel with the bf16 weights (cuBLAS's M=1 bf16
#     kernel reaches ~165 GB/s on GB10, this one ~218); same weights, fp32 accumulation, a different summation order.
#   INKLING_ATTN_FP8=1: qkvr and wo_ud stored as e4m3 with a per-row scale (half the bytes), via gb10_fp8.linear.
p = os.path.join(d, "ops", "linear.py"); s = open(p).read()
if "_ATTN_GEMV" not in s:
    s = s.replace('''    batches over LOW_M_MAX_TOKENS only)."""\n''',
                  '''    batches over LOW_M_MAX_TOKENS only)."""\n'''
                  "    if weight.dtype == torch.float8_e4m3fn:\n"
                  "        from .gb10_fp8 import linear as _linear_fp8\n"
                  "        return _linear_fp8(x, weight, out)\n"
                  "    if _ATTN_GEMV and out is None and x.shape[0] <= 16 and weight.stride(1) == 1:\n"
                  "        from .gb10_fp8 import gemv as _gemv\n"
                  "        return _gemv(x.contiguous(), weight, None)\n", 1)
    s = s.replace("LOW_M_MAX_TOKENS = 32\n", "LOW_M_MAX_TOKENS = 32\nimport os as _os\n"
                  "_ATTN_GEMV = _os.environ.get('INKLING_ATTN_GEMV') == '1'\n", 1)
    assert s.count("_ATTN_GEMV") == 3, s.count("_ATTN_GEMV")
    open(p, "w").write(s)
p = os.path.join(d, "attention.py"); s = open(p).read()
if "_ATTN_FP8" not in s:
    s = s.replace("        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)\n",
                  "        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)\n"
                  "        import os as _os  # _ATTN_FP8\n"
                  "        if _os.environ.get('INKLING_ATTN_FP8') == '1':\n"
                  "            from .ops.gb10_fp8 import quantize\n"
                  "            for lin in (self.qkvr, self.wo_ud):\n"
                  "                w8, _ = quantize(lin.weight.data)\n"
                  "                lin.weight.data = w8\n"
                  "            torch.cuda.empty_cache()\n", 1)
    assert s.count("_ATTN_FP8") == 2 and "INKLING_ATTN_FP8" in s
    open(p, "w").write(s)
# model.py: INKLING_WEIGHT_OVERLAY=<file.safetensors> loads that file's tensors after the checkpoint's, through the
# same renames and shard loaders (later writes win). Used for Tess 4.5: its LoRA merged into the BF16 attention and
# unembed (tess/merge.py), 10.5 GiB instead of a second 367 GiB checkpoint.
p = os.path.join(d, "model.py"); s = open(p).read()
if "OVERLAY_HOOK" not in s:
    s = s.replace("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
                  "        params = dict(self.named_parameters())\n",
                  "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
                  "        import itertools as _it, os as _os  # OVERLAY_HOOK\n"
                  "        _overlay = _os.environ.get('INKLING_WEIGHT_OVERLAY')\n"
                  "        if _overlay:\n"
                  "            def _overlay_weights():\n"
                  "                from safetensors import safe_open\n"
                  "                with safe_open(_overlay, 'pt', device='cpu') as f:\n"
                  "                    for k in f.keys():\n"
                  "                        yield k, f.get_tensor(k)\n"
                  "            weights = _it.chain(weights, _overlay_weights())\n"
                  "        params = dict(self.named_parameters())\n", 1)
    assert s.count("OVERLAY_HOOK") == 1
    open(p, "w").write(s)
# moe.py: INKLING_EXPERT_STATS=<path.npy>: rank 0 counts each layer's routed experts (gb10 expert_stats.py).
p = os.path.join(d, "moe.py"); s = open(p).read()
if "STATS_HOOK" not in s:
    s = s.replace("        self.gate = Inkling2Gate(config)\n        self.experts = Inkling2Experts(config, quant_format)\n",
                  "        self.gate = Inkling2Gate(config)\n        self.experts = Inkling2Experts(config, quant_format)\n"
                  "        from .ops import expert_stats as _es  # STATS_HOOK\n"
                  "        self.stats_layer = _es.register(config.n_routed_experts, get_tensor_model_parallel_rank())\n", 1)
    s = s.replace("        routing, quantized = route(\n",
                  "        if self.stats_layer is not None:\n"
                  "            from .ops import expert_stats as _es\n"
                  "            _es.record(self.stats_layer, x, gate.weight, gate.bias, gate.topk, is_padding)\n"
                  "        routing, quantized = route(\n", 1)
    assert s.count("STATS_HOOK") == 1 and s.count("_es.record(") == 1
    open(p, "w").write(s)
print("patched", d)
