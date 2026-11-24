"""gb10_fp8 on one GB10: accuracy against bf16 x @ deq(w8).T in fp32, the quantization error itself (vs bf16
weights), row-slice scale lookup, and decode speed against torch's bf16 linear at Inkling's TP=4 shapes.
"""
import itertools
import time

import torch

import gb10_fp8 as f

torch.manual_seed(0)
dev = "cuda"
H = 4608


def timeit(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


# shapes per rank at TP=4: fused qkvr (~3.6K rows x 4608) and wo (4608 x 1536); 8 copies so DRAM, not L2, serves
for name, n, k in (("qkvr", 3600, H), ("wo", H, 1536)):
    ws = [(torch.randn(n, k, device=dev) * 0.02).bfloat16() for _ in range(8)]
    q = [f.quantize(w) for w in ws]
    for m in (1, 4, 16, 64):
        x = torch.randn(m, k, device=dev).bfloat16()
        ref_deq = x.float() @ (q[0][0].float() * q[0][1][:, None]).T
        ref_bf = x.float() @ ws[0].float().T
        y = f.linear(x, q[0][0])
        e_kernel = ((y.float() - ref_deq).norm() / ref_deq.norm()).item()
        e_quant = ((ref_deq - ref_bf).norm() / ref_bf.norm()).item()
        line = f"{name} M={m:3d}: kernel vs deq ref {e_kernel:.2e} | fp8 vs bf16 weights {e_quant:.2e}"
        if m <= 16:
            i = [0]

            def run_fp8():
                i[0] = (i[0] + 1) % 8
                f.linear(x, q[i[0]][0])

            def run_bf():
                i[0] = (i[0] + 1) % 8
                torch.nn.functional.linear(x, ws[i[0]])
            t8, tb = timeit(run_fp8), timeit(run_bf)
            line += f" | fp8 {t8:6.1f} us ({n * k / t8 / 1e3:5.0f} GB/s) vs bf16 {tb:6.1f} us ({2 * n * k / tb / 1e3:5.0f} GB/s)"
        print(line)
    # row slice: scales follow the view
    a, b = 1000, 1500
    x = torch.randn(1, k, device=dev).bfloat16()
    ys = f.linear(x, q[0][0][a:b])
    yf = f.linear(x, q[0][0])[:, a:b]
    print(f"{name} slice [{a}:{b}] equals full[:, {a}:{b}]: {torch.equal(ys, yf)}")

# best decode config for each shape
for name, n, k in (("qkvr", 3600, H), ("wo", H, 1536)):
    ws = [f.quantize((torch.randn(n, k, device=dev) * 0.02).bfloat16()) for _ in range(8)]
    x = torch.randn(1, k, device=dev).bfloat16()
    best = None
    for bn, bk, warps, stages in itertools.product((16, 32, 64, 128), (128, 256, 512), (2, 4, 8), (2, 3, 4)):
        i = [0]

        def run():
            i[0] = (i[0] + 1) % 8
            f.gemv(x, ws[i[0]][0], ws[i[0]][1], bn=bn, bk=bk, warps=warps, stages=stages)
        try:
            t = timeit(run, 50)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, (bn, bk, warps, stages))
    print(f"{name} best M=1 config {best[1]}: {best[0]:.1f} us ({n * k / best[0] / 1e3:.0f} GB/s)")
