"""Expert cache for serving Inkling 2 Small on two Sparks (INKLING_EXPERT_CACHE).

Each MoE layer keeps `slots` experts (this rank's half of each) in GPU memory instead of all 384, in two independent
caches: one for the gate/up projection (w13 + its scales) and one for the down projection (w2 + its scales), each with
its own slot map read by the GEMM kernels. The rest live in the rank's expert pack on NVMe (two-spark/pack_experts.py).

Per step, after routing (the tile table lists each row tile's expert): if the step's experts fit in the slots, both
caches load what is missing (least recently used slots not needed this step are evicted) and the layer runs as usual.
If they do not fit (a long prompt chunk reaches nearly every expert), the layer runs each projection in groups of experts
that fit: tiles of other experts are masked to empty row ranges, so each pass writes only its group's rows into one
shared output; the gate/up pass and the down pass load only their own half of each expert. Routing, scaling and the
arithmetic are unchanged: a resident expert's bytes are exactly the checkpoint's.

Environment:
  INKLING_EXPERT_CACHE=<slots per layer>     e.g. 176
  INKLING_EXPERT_PACK=<dir>                  pack root (index.json, rank{r}/layer{LL}.bin)
  INKLING_EXPERT_STATS_FILE=<npy>            optional [48, 384] router counts: the initial set is each layer's most used
  INKLING_EXPERT_CACHE_THREADS=<n>           pread threads (default 32)
  INKLING_EXPERT_DIRECT=0|1                  O_DIRECT reads (default 1)
  INKLING_EXPERT_PREFETCH=0|1                prefetch the next layer's predicted experts (default 0)
  INKLING_EXPERT_PREFETCH_TOKENS=<n>         only for steps of at most n tokens (default 16: decode)

Prefetch: after layer L routes, layer L+1's router is applied to layer L's MoE input (the residual stream changes little
between layers) and the predicted experts are loaded into layer L+1's cache by a background thread on its own CUDA
stream, while layer L's GEMMs and layer L+1's attention run. Layer L+1 waits for that load before its own (exact) misses,
so only one load is ever in flight and the shared staging is never used twice at once. A wrong guess costs bandwidth and
an evicted slot, never correctness: layer L+1 still loads whatever routing actually picked.
"""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import torch

_POOL = None
_POOL_LOCK = threading.Lock()
STATS = {"hits": 0, "misses": 0, "bytes": 0, "load_s": 0.0, "steps": 0, "grouped_steps": 0,
         "pf_loaded": 0, "pf_useful": 0, "pf_s": 0.0, "pf_wait_s": 0.0}
CACHES = {}  # layer -> ExpertCache
PREFETCH = os.environ.get("INKLING_EXPERT_PREFETCH", "0") == "1"  # off: measured slower (7.4 vs 9.2 tok/s, Nov 25)
PREFETCH_TOKENS = int(os.environ.get("INKLING_EXPERT_PREFETCH_TOKENS", "16"))
_PF = {"pool": None, "stream": None}
_STAGING = {}  # (format, piece) -> pinned [batch, bytes], shared by every layer (layers run one at a time)
PIECES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
HALVES = (("w13_weight", "w13_weight_scale"), ("w2_weight", "w2_weight_scale"))
BATCH = 32
CHUNK = 1 << 20  # bytes per pread


def enabled() -> bool:
    return int(os.environ.get("INKLING_EXPERT_CACHE", "0") or 0) > 0


def slots() -> int:
    return int(os.environ["INKLING_EXPERT_CACHE"])


def _pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(int(os.environ.get("INKLING_EXPERT_CACHE_THREADS", "32")))
        return _POOL


class _Half:
    """Cache of one projection's pieces (w13 + scale, or w2 + scale) for one layer."""

    def __init__(self, fd, meta, params, pieces, num_experts):
        self.fd, self.meta, self.params, self.pieces = fd, meta, params, pieces
        self.stride = meta["expert_stride"]
        self.nslots = params[pieces[0]].shape[0]
        dev = params[pieces[0]].device
        self.slot_of = [-1] * num_experts
        self.expert_in = [-1] * self.nslots
        self.last_used = [0] * self.nslots
        self.clock = 0
        self.slot_map = torch.zeros(num_experts, dtype=torch.int32, device=dev)
        for k in pieces:
            key = (meta["format"], k)
            if key not in _STAGING:
                _STAGING[key] = torch.empty((BATCH, meta["pieces"][k]["bytes"]), dtype=torch.uint8, device="cpu", pin_memory=True)
        self.staging = {k: _STAGING[(meta["format"], k)] for k in pieces}
        for k in pieces:
            assert self.staging[k].data_ptr() % 4096 == 0, "pinned staging must be page-aligned for O_DIRECT"

    def _pread(self, k, e, row, lo, hi):
        """Bytes [lo, hi) of piece k of expert e into staging row `row`."""
        pc = self.meta["pieces"][k]
        mv = memoryview(self.staging[k][row].numpy()).cast("B")[lo:hi]
        n = os.preadv(self.fd, [mv], e * self.stride + pc["offset_in_expert"] + lo)
        assert n == hi - lo, (k, e, lo, n)

    def _load(self, pairs, stat="load_s"):
        # A layer usually misses one or two experts: split every piece into CHUNK-sized reads so even one expert keeps
        # the NVMe busy with many reads in flight (one read per piece ran at ~0.8 GB/s; the drive does ~8 GB/s).
        t0 = time.time()
        for i in range(0, len(pairs), BATCH):
            chunk = pairs[i:i + BATCH]
            futs = [_pool().submit(self._pread, k, e, j, lo, min(lo + CHUNK, self.meta["pieces"][k]["bytes"]))
                    for j, (e, _) in enumerate(chunk) for k in self.pieces
                    for lo in range(0, self.meta["pieces"][k]["bytes"], CHUNK)]
            for f in futs:
                f.result()
            for j, (e, s) in enumerate(chunk):
                for k in self.pieces:
                    self.params[k].data[s].view(torch.uint8).view(-1).copy_(self.staging[k][j], non_blocking=True)
                    STATS["bytes"] += self.meta["pieces"][k]["bytes"]
            torch.cuda.current_stream().synchronize()  # staging is reused by the next chunk
        STATS[stat] += time.time() - t0

    def _push_map(self):
        m = torch.tensor([s if s >= 0 else 0 for s in self.slot_of], dtype=torch.int32, device="cpu")
        self.slot_map.copy_(m, non_blocking=True)

    def fill(self, order):
        pairs = [(int(e), s) for s, e in enumerate(order[: self.nslots])]
        self._load(pairs)
        for e, s in pairs:
            self.slot_of[e], self.expert_in[s] = s, e
        self._push_map()

    def ensure(self, need):
        """Make every expert in `need` (at most nslots) resident; returns the number loaded."""
        self.clock += 1
        miss = [e for e in need if self.slot_of[e] < 0]
        for e in need:
            if self.slot_of[e] >= 0:
                self.last_used[self.slot_of[e]] = self.clock
        if not miss:
            return 0
        assert len(need) <= self.nslots
        needset = set(need)
        victims = sorted((s for s in range(self.nslots) if self.expert_in[s] not in needset), key=lambda s: self.last_used[s])[: len(miss)]
        pairs = list(zip(miss, victims))
        self._load(pairs)
        for e, s in pairs:
            old = self.expert_in[s]
            if old >= 0:
                self.slot_of[old] = -1
            self.slot_of[e], self.expert_in[s], self.last_used[s] = s, e, self.clock
        self._push_map()
        return len(miss)

    def prefetch(self, pred):
        """Load the predicted experts (at most nslots) that are not resident; evicts least recently used slots outside
        `pred`. Returns the experts loaded."""
        miss = [e for e in pred if self.slot_of[e] < 0]
        if not miss:
            return []
        predset = set(pred)
        victims = sorted((s for s in range(self.nslots) if self.expert_in[s] not in predset), key=lambda s: self.last_used[s])[: len(miss)]
        pairs = list(zip(miss, victims))
        self._load(pairs, "pf_s")
        for e, s in pairs:
            old = self.expert_in[s]
            if old >= 0:
                self.slot_of[old] = -1
            self.slot_of[e], self.expert_in[s], self.last_used[s] = s, e, self.clock
        self._push_map()
        return [e for e, _ in pairs]

    def resident_first(self, need):
        return sorted(need, key=lambda e: (self.slot_of[e] < 0, e))


class ExpertCache:
    PIECES = PIECES

    def __init__(self, layer: int, rank: int, params: dict, num_experts: int):
        root = os.environ["INKLING_EXPERT_PACK"]
        meta = json.load(open(os.path.join(root, "index.json")))["layers"][str(layer)]
        assert meta["experts"] == num_experts
        for k, p in params.items():
            assert p[0].numel() * p.element_size() == meta["pieces"][k]["bytes"], (k, tuple(p.shape), meta["pieces"][k])
        self.layer = layer
        path = os.path.join(root, f"rank{rank}", f"layer{layer:02d}.bin")
        # O_DIRECT skips the page cache: a layer's few-MiB burst of reads takes ~1.4 ms instead of ~4 ms (measured on a
        # Spark with the page cache squeezed by the model), and the pack is far bigger than the cache anyway. It needs
        # 512-byte aligned offsets, lengths and buffers: the pack's pieces and strides are, and pinned staging is
        # page-aligned. INKLING_EXPERT_DIRECT=0 reads through the page cache instead.
        direct = os.environ.get("INKLING_EXPERT_DIRECT", "1") == "1" and all(
            (v["offset_in_expert"] | v["bytes"] | meta["expert_stride"]) % 512 == 0 for v in meta["pieces"].values())
        self.fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
        self.direct = direct
        if layer == 0:
            print(f"[expert-cache] rank {rank}: {'O_DIRECT' if direct else 'page-cache'} reads", flush=True)
        self.halves = [_Half(self.fd, meta, params, h, num_experts) for h in HALVES]
        self.nslots = self.halves[0].nslots
        self.pending, self.prefetched = None, set()
        CACHES[layer] = self

    def wait(self):
        """Finish this layer's prefetch (submitted by the previous layer) before touching its slots."""
        if self.pending is not None:
            t0 = time.time()
            self.pending.result(); self.pending = None
            STATS["pf_wait_s"] += time.time() - t0

    def prefetch(self, pred):
        """Start loading `pred` (this layer's predicted experts) in the background."""
        if self.pending is not None or not pred or len(pred) > self.nslots:
            return
        if _PF["pool"] is None:
            _PF["pool"] = ThreadPoolExecutor(1)
            _PF["stream"] = torch.cuda.Stream(device=self.halves[0].slot_map.device)
        def run():
            with torch.cuda.stream(_PF["stream"]):
                got = set(self.halves[0].prefetch(pred))
                self.halves[1].prefetch(pred)
                _PF["stream"].synchronize()
            self.prefetched |= got
            STATS["pf_loaded"] += len(got)
        self.pending = _PF["pool"].submit(run)

    def fill(self, order):
        for h in self.halves:
            h.fill(order)

    @property
    def slot_maps(self):
        return self.halves[0].slot_map, self.halves[1].slot_map

    def experts_of(self, tiles: torch.Tensor, max_ctas: int):
        """Experts of this step, from the tile table (entries [0, n) hold each row tile's expert, n at [2 * max_ctas]):
        the routing kernels zero their per-expert counts, so the tile table is what survives. One sync per layer."""
        tt = tiles[: 2 * max_ctas + 1].cpu()
        n = int(tt[2 * max_ctas])
        return sorted(set(tt[:n].tolist()))

    def ensure(self, need):
        """Both projections resident for `need` (fits in the slots). Counts hits and misses."""
        _report()
        STATS["steps"] += 1
        miss = sum(1 for e in need if self.halves[0].slot_of[e] < 0)
        if self.prefetched:
            STATS["pf_useful"] += len(self.prefetched.intersection(need)); self.prefetched = set()
        STATS["hits"] += len(need) - miss; STATS["misses"] += miss
        for h in self.halves:
            h.ensure(need)

    def groups(self, half: int, need):
        """Split `need` into slot-sized groups, the experts already resident in this projection's cache first."""
        h = self.halves[half]; order = h.resident_first(need)
        return [order[i:i + h.nslots] for i in range(0, len(order), h.nslots)]


_LAST = {"t": None, "snap": None}


def _report(every=480):
    """Every `every` layer-steps (10 tokens at 48 layers), print what the cache did since the last report."""
    if STATS["steps"] % every:
        return
    now = time.time(); snap = dict(STATS)
    if _LAST["t"] is not None:
        d = {k: snap[k] - _LAST["snap"][k] for k in snap}; dt = now - _LAST["t"]
        hit = d["hits"] / max(1, d["hits"] + d["misses"])
        print(f"[expert-cache] {dt:.1f}s for {d['steps']} layer-steps: hit {hit:.1%}, {d['misses']} misses, "
              f"{d['bytes'] / 2**30:.2f} GiB loaded in {d['load_s']:.1f}s ({d['bytes'] / max(1e-9, d['load_s']) / 1e9:.1f} GB/s), "
              f"grouped steps {d['grouped_steps']}; prefetched {d['pf_loaded']} ({d['pf_useful']} used) in {d['pf_s']:.1f}s, "
              f"waited {d['pf_wait_s']:.2f}s", flush=True)
    _LAST["t"], _LAST["snap"] = now, snap


def masked_tiles(tiles: torch.Tensor, max_ctas: int, tile: int, keep: list, num_experts: int) -> torch.Tensor:
    """Copy of the tile table in which tiles whose expert is not in `keep` cover an empty row range ([t*TILE, t*TILE)),
    so a GEMM over it writes only the rows of `keep`'s experts."""
    m = max_ctas
    out = tiles.clone()
    flags = torch.zeros(num_experts, dtype=torch.bool, device=tiles.device)
    flags[torch.tensor(keep, dtype=torch.long, device=tiles.device)] = True
    ids = tiles[:m].long().clamp(0, num_experts - 1)
    starts = torch.arange(m, device=tiles.device, dtype=tiles.dtype) * tile
    out[m:2 * m] = torch.where(flags[ids], tiles[m:2 * m], starts)
    return out


def initial_order(layer: int, num_experts: int):
    """Most-used experts first, from INKLING_EXPERT_STATS_FILE ([layers, experts] counts); else 0..E-1."""
    p = os.environ.get("INKLING_EXPERT_STATS_FILE")
    if p and os.path.exists(p):
        import numpy as np
        c = np.load(p)[layer]
        return list(np.argsort(-c, kind="stable"))
    return list(range(num_experts))
