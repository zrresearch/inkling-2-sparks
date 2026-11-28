"""Write the expert packs for two-Spark serving: per TP rank (0, 1) and MoE layer, one file holding every expert's half in
the layout the GB10 kernel reads (the checkpoint's natural layout; no shuffling or swizzling on GB10), each piece
4 KiB-aligned so one pread loads one piece. Same split as INKLING_MOE_TP at TP=2: w13 rows [r*n, (r+1)*n) (interleaved
gate/up pairs, so contiguous), w2 columns [r*c, (r+1)*c), and the block scales split the same way. Per-expert amax and the
per-projection outer-scale denominators are small and stay resident; they are not packed.
  pack_experts.py CHECKPOINT_DIR OUT_DIR [--layers 0-47]
Writes OUT_DIR/rank{r}/layer{L:02d}.bin and OUT_DIR/index.json (offset and byte size of every piece)."""
import json, os, struct, sys, time, numpy as np
CK, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
LAY = range(48)
if "--layers" in sys.argv:
    a, b = sys.argv[sys.argv.index("--layers") + 1].split("-"); LAY = range(int(a), int(b) + 1)
TP, ALIGN = 2, 4096
wmap = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_h = {}
def tensor(name):
    f = f"{CK}/{wmap[name]}"
    if f not in _h:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (8 + n, json.loads(fh.read(n)))
    off, h = _h[f]; m = h[name]; a, b = m["data_offsets"]
    esz = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}[m["dtype"]]
    shape = m["shape"][:-1] + [m["shape"][-1] * esz]  # view everything as bytes
    return np.memmap(f, dtype=np.uint8, mode="r", offset=off + a, shape=tuple(shape)), m["dtype"], m["shape"]
os.makedirs(OUT, exist_ok=True)
idx_path = f"{OUT}/index.json"
index = json.load(open(idx_path)) if os.path.exists(idx_path) else {"tp": TP, "align": ALIGN, "layers": {}}
pad = lambda n: (n + ALIGN - 1) // ALIGN * ALIGN
t0 = time.time()
for L in LAY:
    if str(L) in index["layers"]: continue
    p = f"model.llm.layers.{L}.mlp.experts."
    T = {k: tensor(p + k) for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")}
    E = T["w13_weight"][0].shape[0]
    pieces = {}
    for k, (arr, dt, shp) in T.items():
        if k.startswith("w13"):
            n = arr.shape[1] // TP; pieces[k] = (lambda arr=arr, n=n: (lambda e, r: arr[e, r * n:(r + 1) * n, :]))(), dt, [n, shp[2]]
        else:
            c = arr.shape[2] // TP; cs = shp[2] // TP; pieces[k] = (lambda arr=arr, c=c: (lambda e, r: arr[e, :, r * c:(r + 1) * c]))(), dt, [shp[1], cs]
    layout, pos = {}, 0
    for k, (_, dt, shp) in pieces.items():
        nb = int(np.prod(shp)) * {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}[dt]
        layout[k] = {"offset_in_expert": pos, "bytes": nb, "dtype": dt, "shape": shp}; pos += pad(nb)
    stride = pos
    for r in range(TP):
        os.makedirs(f"{OUT}/rank{r}", exist_ok=True)
        with open(f"{OUT}/rank{r}/layer{L:02d}.bin", "wb") as f:
            for e in range(E):
                for k, (get, dt, shp) in pieces.items():
                    b = np.ascontiguousarray(get(e, r)).tobytes()
                    assert len(b) == layout[k]["bytes"]
                    f.write(b); f.write(b"\0" * (pad(len(b)) - len(b)))
    index["layers"][str(L)] = {"experts": E, "expert_stride": stride, "pieces": layout,
                               "format": "nvfp4" if T["w13_weight"][1] == "U8" else "mxfp8"}
    json.dump(index, open(idx_path, "w"), indent=1)
    print(f"layer {L:2d} {index['layers'][str(L)]['format']}: {E} experts x {stride / 2**20:.2f} MiB per rank "
          f"({E * stride / 2**30:.1f} GiB), {time.time() - t0:.0f}s", flush=True)
print("PACK DONE")
