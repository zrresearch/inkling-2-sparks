"""Slim checkpoint for two-Spark serving: the Inkling 2 Small checkpoint without the four large per-expert tensors of
every MoE layer (w13_weight, w13_weight_scale, w2_weight, w2_weight_scale), which the expert cache reads from the packs.
Everything else (attention, embeddings, norms, router, per-expert amax and outer-scale denominators, vision, drafter)
is copied byte for byte into new shards with the same tensor names. Config, tokenizer and template files are copied.
  make_slim.py CHECKPOINT_DIR OUT_DIR"""
import json, os, shutil, struct, sys
CK, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DROP = (".mlp.experts.w13_weight", ".mlp.experts.w13_weight_scale", ".mlp.experts.w2_weight", ".mlp.experts.w2_weight_scale")
drop = lambda k: any(k.endswith(x) for x in DROP)
idx = json.load(open(f"{CK}/model.safetensors.index.json"))
new_map, total, dropped = {}, 0, 0
for shard in sorted(set(idx["weight_map"].values())):
    with open(f"{CK}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        keep = [(k, v) for k, v in h.items() if k != "__metadata__" and not drop(k)]
        dropped += sum(1 for k in h if k != "__metadata__" and drop(k))
        if not keep:
            continue
        hdr, off = {}, 0
        if "__metadata__" in h: hdr["__metadata__"] = h["__metadata__"]
        for k, v in keep:
            a, b = v["data_offsets"]; hdr[k] = dict(v, data_offsets=[off, off + b - a]); off += b - a
        hb = json.dumps(hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(f"{OUT}/{shard}", "wb") as o:
            o.write(struct.pack("<Q", len(hb))); o.write(hb)
            for k, v in keep:
                a, b = v["data_offsets"]; f.seek(base + a); left = b - a
                while left:
                    chunk = f.read(min(left, 1 << 26)); o.write(chunk); left -= len(chunk)
                new_map[k] = shard
        total += off
        print(f"{shard}: kept {len(keep)} tensors, {off / 2**30:.2f} GiB", flush=True)
json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, open(f"{OUT}/model.safetensors.index.json", "w"), indent=1)
for fn in os.listdir(CK):
    if not fn.endswith(".safetensors") and fn != "model.safetensors.index.json" and os.path.isfile(f"{CK}/{fn}"):
        shutil.copy(f"{CK}/{fn}", f"{OUT}/{fn}")
print(f"SLIM DONE: {len(new_map)} tensors, {total / 2**30:.1f} GiB; dropped {dropped} expert tensors")
