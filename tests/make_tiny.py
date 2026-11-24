"""Build a truncated Inkling 2 Small checkpoint (first N layers, vision tower kept) for single-Spark vLLM smoke tests."""
import json, os, re, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
def keep(k):
    m = re.search(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", k)
    if m is None or k.startswith("model.visual"):
        return True
    lim = {"layers": n, "attn_queries": n - 1, "mlp_queries": n}[m.group(1)]
    return int(m.group(2)) < lim
by_file = {}
for k, f in wm.items():
    if keep(k):
        by_file.setdefault(f, []).append(k)
tensors = {}
for f, ks in by_file.items():
    with safe_open(f"{src}/{f}", "pt") as h:
        for k in ks:
            tensors[k] = h.get_tensor(k)
save_file(tensors, f"{dst}/model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(f"{src}/config.json"))
t = cfg["text_config"]
t["num_hidden_layers"] = n
t["local_layer_ids"] = [i for i in t["local_layer_ids"] if i < n]
if "mtp_config" in cfg:
    del cfg["mtp_config"]
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for f in os.listdir(src):
    if f.endswith((".json", ".jinja")) and f not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f"{src}/{f}", dst)
if os.path.isdir(f"{src}/tiktoken"):
    shutil.copytree(f"{src}/tiktoken", f"{dst}/tiktoken", dirs_exist_ok=True)
print(len(tensors), "tensors", sum(v.numel() * v.element_size() for v in tensors.values()) / 2**30, "GiB")
