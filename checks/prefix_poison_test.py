"""Repro for the prefix-cache poisoning: a repeated prompt of 256 + 15 tokens (a prefix-cache hit leaving a short
query) turns the server's output into repeated junk for every request sharing cached blocks. Prints PASS/FAIL."""
import inspect, collections, json, sys, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8899"
def post(p, d):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + p, json.dumps(d).encode(), {"content-type": "application/json"}), timeout=900))
def hits():
    try:
        t = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
        return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith("vllm:prefix_cache_hits") and not l.startswith("#"))
    except Exception:
        return -1
def comp(x, n=32):
    h0 = hits(); r = comp1(x, n); HIT.append(int(hits() - h0)); return r
HIT = []
def comp1(x, n=32):
    return post("/v1/completions", {"model": "Inkling-2-Small", "prompt": x, "max_tokens": n, "temperature": 0})["choices"][0]["text"]
def junk(t):
    return any(t.count(t[i:i + k]) * k > 0.6 * len(t) for k in (1, 2, 3, 4) for i in range(min(8, len(t))) if t[i:i + k].strip())
ids = post("/tokenize", {"model": "Inkling-2-Small", "prompt": inspect.getsource(collections)[:20000]})["tokens"]
bad = 0
for L in (168, 134, 271, 400, 401, 414, 527):
    a, b = comp(ids[:L]), comp(ids[:L])
    print(L, L % 128, "hit", HIT[-2], HIT[-1], repr(a[:50]), "| repeat:", repr(b[:50]), flush=True)
    bad += junk(a) + junk(b)
h = post("/v1/chat/completions", {"model": "Inkling-2-Small", "messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}], "max_tokens": 400, "temperature": 0})["choices"][0]["message"].get("content")
print("health:", repr((h or "")[:60]))
print("PASS" if bad == 0 and "391" in (h or "") else "FAIL")
