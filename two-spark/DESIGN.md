# Inkling 2 Small on two Sparks: expert cache with NVMe backing

## Budget

| Per Spark (TP=2) | GiB |
|---|---|
| Unified memory usable | ~120 |
| Non-expert weights: attention (FP8, half of 17.8 GiB BF16), embeddings, unembed, norms, router | ~12 |
| Context cache (target 2 streams x 64K, or 1 x 128K) | ~8 |
| vLLM/CUDA overhead, activations, staging | ~8 |
| **Expert cache** | **~85-90** |
| Each Spark's half of every expert, on its NVMe | 175 |

So each Spark can hold about half of its 175 GiB of expert halves in memory. From the router counts of 15.1M Tier 2 tokens,
keeping each layer's ~190 most-used experts covers 90.6% of router picks in the median layer (82% in the worst).

## Speed estimate

- Per token, 48 layers x 8 picks = 384 expert picks; at 90% hits, ~38 misses.
- One expert half on one Spark: 3 x 4608 x 1152 x 0.5625 B (NVFP4 incl. scales) = ~9 MB (MXFP8 layers ~16 MB).
- ~38 x ~10 MB = ~0.4 GB per token per Spark; at 8.2 GB/s (32-thread pread) = ~50 ms.
- Compute + exchange at TP=2: ~25-30 ms per token (TP4 measured 33 ms at 30 tok/s; fewer exchanges, twice the
  weight reads per Spark for the resident part).
- Misses are known only after each layer's router runs, so loads serialize per layer: ~48 small batches of reads per
  token. Expected: **~10-13 tok/s one stream** at 90% hits; less with many streams (more distinct experts per step).
- A cache that keeps recently used experts (LRU over the static top set) should beat the static coverage: a
  conversation reuses experts.

## Pieces

1. **Expert pack writer** (`pack_experts.py`, DONE: verified byte-identical to checkpoint slices; 8.54 MiB per NVFP4
   expert half, 15.66 MiB per MXFP8 half): from the checkpoint, write per rank (0, 1) and layer one file of expert
   halves in the kernel's layout (w13 rows `r*n:(r+1)*n`, w2 columns `r*c:(r+1)*c`, and their scales), each expert at a
   4 KiB-aligned offset so one `pread` loads one expert half. ~175 GiB per rank, on each Spark's NVMe.
2. **Loader change** (`patch_vllm.py`, `INKLING_EXPERT_CACHE=<slots per layer>`): register expert buffers with
   `slots` rows instead of 384; load only the hot set at start (from `expert-usage.npy`); keep a per-layer
   `expert -> slot` table (CPU and GPU copies) and LRU state.
3. **Forward change** (MoE layer): after routing, copy the step's top-k ids to the CPU (one sync per layer), find
   misses, evict LRU slots, `pread` misses with a 32-thread pool into a pinned staging area, `cudaMemcpyAsync` into the
   slots, update the table, remap ids to slots, then run `gb10_bmm` unchanged.
4. **Serving**: TP=2 over the two Sparks; eager mode for a first version (the MoE step now syncs to the CPU, which CUDA
   graphs can't capture; eager mode cost 5% on four Sparks). Prefix caching stays on (the corruption came from
   piecewise graphs, which eager mode does not use).
5. **Checks**: WikiText NLL against the four-Spark tuned build (must be identical up to the TP-split rounding: same
   arithmetic, experts loaded from disk instead of memory); the prefix-cache repro; speed.py; hit-rate counters per
   layer exported in /metrics.

## Open questions

- ~~Whether GB10's CPU can `pread` straight into a `cudaMalloc` slot~~ Measured: no, `pread` into
  `cudaMalloc` memory fails with EFAULT. `pread` into pinned host memory runs at 8.2 GB/s (32 threads, 9 MiB pieces) and the
  pinned-to-device copy at 58 GB/s, so the staging copy adds ~15% to a miss.
- Real hit rates on agent traffic (long code contexts) versus the Tier 2 counts.
- Whether to split misses across both Sparks' NVMe at once (they read in parallel already: each loads its own half).

## Progress

  experts in shuffled slots, NVFP4 and MXFP8, both projections at TP=2 shapes, 1/8/64/512 tokens.

## First working version

- Correct answers; WikiText NLL 3.0691 (4-Spark reference 3.0697, -0.02%; 4-Spark tuned 3.0732), top-1 agreement 95.2-95.4%.
- Fixes on the way: staging must be explicitly on CPU (model built under a CUDA default device); the loader must accept
  pack-filled expert tensors; a long prompt step needs ~380 experts > slots, so such steps run each projection in
  slot-sized groups (masked tile tables, shared output); separate caches for w13 and w2 so each pass loads only its half.
- Decode, 1 stream: 2.4 tok/s with one pread per piece (0.8 GB/s), 4.1 tok/s with 1 MiB parallel chunks (1.7 GB/s).
  Hit rate 94-96%, ~20 misses (~170 MB) per token. Non-load time ~90 ms/token (eager mode, 48 CPU syncs per token).
- Prompt reading: 17.5K tokens in 144 s (~120 tok/s) - grouped loads dominate.
- Next: overlap loads with compute (load w2 halves while w13/SwiGLU run; prefetch), more slots (KV can shrink),
  larger prompt chunks to amortize grouped loads, O_DIRECT reads; then fold the dev files into the image (patch_cache.py).

## O_DIRECT expert loads (4.1 -> 9.2 tok/s)
- The NVMe does 9-10 GB/s on big random reads, but a layer's typical miss burst (~9 MiB) took ~4 ms through the page
  cache (squeezed to ~2 GB free by the model) vs 1.4 ms with O_DIRECT. Pack offsets, lengths and strides are all
  512-byte multiples (logical block size 512) and pinned staging is page-aligned, so O_DIRECT needs no layout change.
- [expert-cache] per 10 tokens: 0.9-1.5 s (was ~2.0 s); loads 0.4-0.8 s at ~4 GB/s (was 1.1 s at 1.7 GB/s).
- Single stream, 48-token prompt, 300 tokens (speedcheck.py): 9.10-9.19 tok/s warm (was 4.1). Output coherent.
- Next-layer prefetch (layer L+1's router on layer L's input, background thread + side stream): implemented, measured
  SLOWER (7.4 tok/s): only ~60% of prefetched experts were used, wasted loads evict useful ones, and the loader thread
  competes for the GIL with eager-mode kernel launches. Off by default (INKLING_EXPERT_PREFETCH=1 to try).
- Remaining ideas: more slots (each slot costs ~467 MiB over 48 layers per rank; KV is 18 GiB = 281K tokens now),
  prefetch only high-confidence experts, move the loader out of Python (GIL), CUDA graphs for decode.
