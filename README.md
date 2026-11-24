# Inkling 2 Small on four DGX Sparks

Thinking Machines' [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B mixture of experts,
17B active, NVFP4, 366 GiB) served on **four DGX Sparks (GB10)** with vLLM, 256K context, 16 streams.

Thinking Machines' vLLM image targets B200-class GPUs. On GB10 its expert kernels do not build ("No supported CUDA
architectures found for major versions [10]") and several other kernels need hardware GB10 lacks (tensor memory,
NVLink multicast, more than 99 KB of shared memory per block). This repo patches that image so the model runs on GB10,
and adds three decode speed changes.

Model page on Hugging Face: [umzcio/Inkling-2-Small-DGX-Spark](https://huggingface.co/umzcio/Inkling-2-Small-DGX-Spark).

## Results (four Sparks, tuned build)

**Decode**, tokens per second (one stream: that stream's rate; others: total output including prompt time):

| Build | 1 stream | 4 streams | 8 streams | 16 streams | WikiText perplexity |
|---|---|---|---|---|---|
| Reference (exact arithmetic) | 25.4 | 56.5 | 81.4 | 119.9 | 3.0697 |
| Tuned, recommended settings | 30.2 | 58.5 | 84.0 | 121.0 | 3.0732 (+0.11%) |

**Prompt reading and decode at long context**, one request on an idle server:

| Prompt length | Prompt time | Prompt tok/s | Decode tok/s at that length |
|---|---|---|---|
| 4,096 | 2.2 s | 1,885 | 24.2 |
| 32,768 | 20.0 s | 1,642 | 13.2 |
| 131,072 | 108.3 s | 1,211 | 5.3 |

**Quality**, our Tier 2 suite (temperature 0.6, replies capped at 32,768 tokens; requests that timed out count as wrong):

| Test | Reference build | Tuned build |
|---|---|---|
| GPQA Diamond (198) | 87.4 | 88.4 |
| MMLU-Pro (fixed 700) | 88.1 | 87.6 |
| AIME 2025 (30 × 4) | 93.3 | 90.0 |

The tuned build is within run-to-run spread of the reference on all three. Full reference-build results (IFEval,
LiveBench coding, BFCL, RULER) and the method are in [zrresearch/evals](https://github.com/zrresearch/evals).

**Agent benchmarks** (tuned build, four tasks at a time while other work shared the server):

| Test | Official limits | Longer limits |
|---|---|---|
| Terminal-Bench 2.1, Terminus 2 | 16 / 89 (18.0%) | 27 / 89 (30.3%) at 6× |
| Cybench, Codex, 3 attempts | 17 / 40 tasks (45 min) | 25 / 40 tasks (135 min) |

One 6× task that failed to start (a harness tmux error, not the model) was rerun with the same settings and solved. At official limits most tasks run out of time: with the server shared, each stream decodes at about 8 to 11 tokens per
second, and decode slows further as an agent's context grows (table above).

## What the patch changes

`build/patch_vllm.py` edits vLLM's Inkling code inside Thinking Machines' image; `build/Dockerfile` builds the result.

- **Expert matrix multiplies** (`gb10_bmm.py`): a Triton kernel on Triton's block-scaled dot product that reads the
  checkpoint's weight layout directly, for NVFP4 (layers 0-39) and MXFP8 (layers 40-47).
- **Attention** (`gb10_attention.py`): a Triton paged attention kernel with Inkling's relative position bias, replacing
  the prefill kernel and the tensor-memory decode kernel. It reads only keys inside the sliding window (vLLM points
  pages that slid out of the window at a shared page that can hold NaN) and reads vLLM's bias table in place, which
  makes CUDA graphs work.
- **Small linear layers**: the router layer moves off a path that needs 192 KB of shared memory; small BF16 layers move
  from a FlashInfer kernel that refuses GB10 to cuBLAS.
- **Off**: the vision encoder (B200-only attention kernel; text only) and the DSpark draft model (needs NVLink multicast).

**Speed switches** (environment variables, all off by default = the reference build):

| Variable | What it does | Effect on results |
|---|---|---|
| `INKLING_ATTN_GEMV=1` | Triton matrix-vector kernel for attention weights at small batch (`gb10_fp8.py`); ~230 GB/s against cuBLAS's ~158 | none (bit-identical) |
| `INKLING_MOE_TP=1` | each Spark holds a quarter of every NVFP4 expert instead of a quarter of the experts, so no Spark waits on the busiest one | perplexity +0.09% |
| `INKLING_ATTN_FP8=1` | attention weights in FP8 with one scale per row: half the bytes read per token | perplexity +0.02% more |

Also available: `INKLING_WEIGHT_OVERLAY=<file.safetensors>` loads extra tensors over the checkpoint (we used it to
serve a merged fine-tune, Tess 4.5, from the same expert weights), and `INKLING_EXPERT_STATS=<file.npy>` counts which
experts the router picks per layer.

## Setup

Hardware: four DGX Sparks on a RoCE fabric (we used a QSFP switch), the full checkpoint
on every node at the same path, Docker with the NVIDIA runtime.

```bash
# 1. on every node: the checkpoint
hf download thinkingmachines/Inkling-2-Small --revision e242a97e271095c11bc33ddb3850d37e22917343 --local-dir /path/to/Inkling-2-Small

# 2. on every node: the image (or build once and docker save | docker load)
cd build && docker build -t inkling2-gb10 .

# 3. on the head node
cd serve && cp nodes.conf.example nodes.conf   # fill in addresses, RoCE devices, interface
./serve4.sh start
./serve4.sh logs        # loads in 10 to 13 minutes; leaves room for about 380,000 tokens of cache
```

`serve4.sh` starts the tuned build with the recommended settings. For the reference build: `EXTRA_ENV="" ./serve4.sh start`.

## Known faults and their fixes

**Prefix caching corrupts replies with vLLM's default CUDA graphs.** With the default `cudagraph_mode`
(`FULL_AND_PIECEWISE`), steps that mix a short prompt remainder after a prefix-cache hit (under ~32 tokens) with other
requests' decode tokens, replayed from piecewise CUDA graphs, corrupt cached state: replies turn into runs of "!" or other
repeated junk until they finish, and a hit can also change a reply without visible junk. On AIME, which asks each
problem four times, 59 to 64 of 120 replies broke. Eager mode, or `--compilation-config.cudagraph_mode=FULL_DECODE_ONLY`
(graphs only for decode-only steps), fixes it; `serve4.sh` uses the latter (costs 4% of one-stream decode). We did not
pin which buffer the replay leaves stale; Inkling's short convolution, which keeps a four-token state per request in
the cache, is the likely place. Check any build with `checks/prefix_poison_test.py` (prints PASS or FAIL).

**Splitting the MXFP8 experts across Sparks breaks them.** A quarter of an MXFP8 expert is 576 values wide, which gives
18 scale columns, and the scale layout needs a multiple of 4. `INKLING_MOE_TP` therefore splits only the NVFP4 layers
(0-39) and keeps the MXFP8 layers (40-47) whole per Spark.

## Checks

All take a server URL (default `http://localhost:8899`).

- `checks/nll.py`: log-likelihood of a fixed text, position by position; compare two builds (perplexity, top-1
  agreement). Takes two minutes and caught a bug the replies did not show.
- `checks/golden.py`: greedy replies with logprobs for five fixed prompts, compared across builds.
- `checks/prefix_poison_test.py`: the prefix-cache fault repro above.
- `checks/speed.py`, `checks/ctx_speed.py`: the decode and long-context tables above. `ctx_speed.py` (and our `nll.py`
  runs) read a plain-text file `wiki16k.txt` from the current directory; we used about 16K tokens of WikiText.
- `tests/`: kernel tests against PyTorch references (some need `tests/make_tiny.py`, which cuts a few-layer checkpoint
  for single-Spark tests).

## Limits

Text only; no speculative decoding (DSpark needs NVLink multicast); four Sparks only (the checkpoint does not fit in
two Sparks' memory; a two-Spark setup is future work). Decode at long context is limited by our attention kernel in
the 8 global layers, which is not yet tuned.

## License and credits

Code in this repo: Apache-2.0 (see `LICENSE`). It patches vLLM (Apache-2.0) inside Thinking Machines' image, which is
not redistributed here. Inkling 2 Small is released by Thinking Machines under Apache-2.0.

Thanks to Thinking Machines for the model and its vLLM image, and to the vLLM and Triton projects.

Write-up: zrresearch.org (coming).
