# Inkling 2 Small on DGX Sparks (four, or two)

Thinking Machines' [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B mixture of experts,
17B active, NVFP4, 366 GiB) served on **four DGX Sparks (GB10)** with vLLM, 256K context, 16 streams.

Thinking Machines' vLLM image targets B200-class GPUs. On GB10 its expert kernels do not build ("No supported CUDA
architectures found for major versions [10]") and several other kernels need hardware GB10 lacks (tensor memory,
NVLink multicast, more than 99 KB of shared memory per block). This repo patches that image so the model runs on GB10,
and adds three decode speed changes.

A two-Spark build serves the same checkpoint on two Sparks at about 9 tokens/s for one user, by keeping the most-used
experts in memory and reading the rest from each Spark's NVMe (see [Two Sparks](#two-sparks)).

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

## Two Sparks

The checkpoint is 366 GiB and two Sparks have about 240 GiB of memory, so the two-Spark build does not hold every
expert in memory. Each Spark keeps its half of every non-expert weight, plus 176 of the 384 experts per layer (its half
of each), and reads the rest from its own NVMe when the router picks them. The kept set starts from the experts our
test suite used most and changes as the conversation goes (least recently used out).

**Results** (two Sparks, one direct QSFP cable):

| | Two Sparks | Four Sparks (tuned) |
|---|---|---|
| Decode, 1 stream | 9.2 tok/s | 30.2 tok/s |
| Decode, 4 streams (total) | about 7 tok/s | 58.5 tok/s |
| WikiText perplexity | 3.0691 | 3.0732 (reference build 3.0697) |
| Context | 128K, about 280K tokens of cache | 256K, about 380K tokens of cache |

- One stream: 300 tokens after a 48-token prompt, warm cache. The first request after start runs at about 8.4 tok/s
  while the cache fills.
- Four streams: a 90-minute run with four users sending short questions, essays, and 8K and 32K-token prompts with a
  planted fact to recall: 209 requests, none failed, 60/60 recalls right, 7.1 tok/s in total. More users share the
  same disk reads, so total speed does not grow with streams. **This is a one-user build.**
- Long prompts are slow: with four streams active, a 32K-token prompt took about two minutes to answer.
- Perplexity matches the four-Spark reference build to 0.02%: the arithmetic is the same, only where the expert
  weights come from changes. We have not rerun the Tier 2 quality suite on two Sparks.

**How it works**

- Experts are split across the two Sparks (`INKLING_MOE_TP` at TP=2, all 48 layers), so each Spark reads only its half
  of a missing expert: about 9 MiB (NVFP4 layers) or 16 MiB (MXFP8 layers).
- After each layer's router runs, the missing experts are read from that Spark's NVMe with O_DIRECT into pinned memory
  and copied into free slots. Bypassing the page cache made a layer's typical reads 3x faster (1.4 ms instead of
  4 ms), which took decode from 4.1 to 9.2 tok/s.
- Prompt steps that need more experts than there are slots run in groups of slot size.
- The cache syncs to the CPU in every MoE layer, so the server runs in eager mode (no CUDA graphs).
- Reading the next layer's experts ahead of time (`INKLING_EXPERT_PREFETCH=1`) is implemented but slower (7.4 tok/s):
  only about 60% of the guesses were used. It is off by default.

**Setup**

Hardware: two DGX Sparks joined by one QSFP cable (RoCE), NVMe with fast random reads (ours read 9-10 GB/s), about
190 GiB free on each Spark, Docker with the NVIDIA runtime.

```bash
# once, on a machine with the full checkpoint (366 GiB): the expert packs and the slim checkpoint
python3 two-spark/pack_experts.py /path/to/Inkling-2-Small /path/to/packs      # rank0/ + rank1/, ~175 GiB each
python3 two-spark/make_slim.py   /path/to/Inkling-2-Small /path/to/slim        # 13 GiB: everything except expert weights
# into the same folder (BASE in nodes2.conf) on both Sparks: slim/, packs/index.json, two-spark/expert-usage.npy;
# packs/rank0 on the head, packs/rank1 on the worker

# both Sparks: the four-Spark image, then the two-Spark layer on top
(cd build && docker build -t inkling2-gb10 .)
(cd two-spark && docker build -t inkling2-gb10:tp2 .)

# head Spark
cd two-spark && cp nodes2.conf.example nodes2.conf   # paths, worker address, RoCE devices, interfaces
./serve2.sh start
./serve2.sh logs
```

`SLOTS` (default 176) sets experts kept per layer; each extra slot costs about 467 MiB per Spark across the 48 layers
and comes out of the context cache.

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

Text only; no speculative decoding (DSpark needs NVLink multicast); on two Sparks, see [Two Sparks](#two-sparks): about 9
tokens/s for one user, and it does not scale with more users. Decode at long context is limited by our attention kernel in
the 8 global layers, which is not yet tuned.

## License and credits

Code in this repo: Apache-2.0 (see `LICENSE`). It patches vLLM (Apache-2.0) inside Thinking Machines' image, which is
not redistributed here. Inkling 2 Small is released by Thinking Machines under Apache-2.0.

Thanks to Thinking Machines for the model and its vLLM image, and to the vLLM and Triton projects.

Write-up: zrresearch.org (coming).
