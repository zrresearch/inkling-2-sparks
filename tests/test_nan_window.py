# Does a NaN in a page outside the sliding window (vLLM's never-written null block) leak into the output?
import torch, sys
sys.path.insert(0, "/w")
from gb10_attention import rel_attention
torch.manual_seed(0)
H, KVH, D, PAGE, W, REL = 8, 1, 128, 16, 1024, 512
kv_len = 3000; q_len = 1
nb = 400
kc = torch.randn(nb, PAGE, KVH, D, device="cuda", dtype=torch.bfloat16)
vc = torch.randn(nb, PAGE, KVH, D, device="cuda", dtype=torch.bfloat16)
npages = (kv_len + PAGE - 1) // PAGE
bt = torch.arange(1, npages + 1, device="cuda", dtype=torch.int32)[None].clone()
q = torch.randn(q_len, H, D, device="cuda", dtype=torch.bfloat16)
rel = torch.zeros(q_len + 128, H, REL + 256, device="cuda", dtype=torch.float32)
args = dict(block_table=bt, cache_seqlens=torch.tensor([kv_len], device="cuda", dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, q_len], device="cuda", dtype=torch.int32), max_seqlen_q=q_len,
            softmax_scale=D ** -0.5, window_size=(W - 1, 0), sheared=rel)
def run():
    out = torch.empty(q_len, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.empty(H, q_len, device="cuda", dtype=torch.float32)
    rel_attention(q, kc, vc, out=out, lse=lse, **args); torch.cuda.synchronize(); return out.float()
ref = run()
# vLLM's sliding-window manager points pages wholly before the window at the null block (id 0); fill it with NaN
first_needed = (kv_len - 1 - (W - 1)) // PAGE
bt[0, :first_needed] = 0
kc[0] = float("nan"); vc[0] = float("nan")
out = run()
print("pages replaced by null block:", first_needed, "| NaN in output:", bool(out.isnan().any()),
      "| max diff vs clean:", (out - ref).abs().nan_to_num(1e9).max().item())
