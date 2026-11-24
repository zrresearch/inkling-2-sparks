"""Prompt log-likelihood of a fixed text through a running server, to compare builds (quantization, kernels).

    nll.py save OUT.json TEXT.txt [--url URL] [--chunks 4]
    nll.py compare A.json B.json

save: the text in `chunks` equal pieces, each one prompt with prompt_logprobs=1; records, per position, the
logprob of the actual token and the model's top-1 token. compare: perplexity of each, top-1 agreement between the
two, and the mean |logprob difference| of the actual tokens.
"""
import json, math, sys, urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def save(out, text_path, url, chunks):
    text = open(text_path).read()
    n = len(text) // chunks
    rec = []
    for c in range(chunks):
        piece = text[c * n:(c + 1) * n]
        j = post(f"{url}/v1/completions", {"model": "Inkling-2-Small", "prompt": piece, "max_tokens": 1,
                                           "temperature": 0, "prompt_logprobs": 1})
        pl = j["choices"][0]["prompt_logprobs"]
        for pos in pl[1:]:  # first position has no prediction
            items = list(pos.items())
            actual = items[0]  # vLLM lists the prompt token first, then the top-k
            top = min(items, key=lambda kv: kv[1]["rank"])
            rec.append((actual[1]["logprob"], int(top[0])))
        print(f"chunk {c}: {len(pl) - 1} positions", flush=True)
    json.dump(rec, open(out, "w"))
    nll = -sum(r[0] for r in rec) / len(rec)
    print(f"{len(rec)} positions, perplexity {math.exp(nll):.4f}")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    n = min(len(A), len(B))
    ppl = lambda R: math.exp(-sum(r[0] for r in R[:n]) / n)
    agree = sum(A[i][1] == B[i][1] for i in range(n)) / n
    d = sum(abs(A[i][0] - B[i][0]) for i in range(n)) / n
    print(f"{n} positions: perplexity {ppl(A):.4f} -> {ppl(B):.4f} ({100 * (ppl(B) / ppl(A) - 1):+.3f}%), "
          f"top-1 agreement {100 * agree:.2f}%, mean |dlogprob| {d:.4f}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "save":
        url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:8899"
        ch = int(a[a.index("--chunks") + 1]) if "--chunks" in a else 4
        save(a[2], a[3], url, ch)
    else:
        compare(a[2], a[3])
