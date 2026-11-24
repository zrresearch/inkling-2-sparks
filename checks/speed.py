"""Decode speed of a running server: N concurrent streams, each a fixed prompt and `tokens` tokens (ignore_eos,
greedy), no logprobs. Prints total tokens/s and per-stream tokens/s (from first to last token) for each N.

    speed.py [--url URL] [--tokens 512] [--streams 1,4,8,16] [--model Inkling-2-Small]
"""
import json, sys, threading, time, urllib.request

a = sys.argv
URL = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:8899"
TOK = int(a[a.index("--tokens") + 1]) if "--tokens" in a else 512
STREAMS = [int(x) for x in (a[a.index("--streams") + 1] if "--streams" in a else "1,4,8,16").split(",")]
MODEL = a[a.index("--model") + 1] if "--model" in a else "Inkling-2-Small"
PROMPT = "Write a detailed technical explanation of how a hash map handles collisions, with Python examples."


def one(i, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"{PROMPT} (variant {i})"}],
            "max_tokens": TOK, "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("usage"):
                n = j["usage"]["completion_tokens"]
            if j.get("choices"):
                now = time.time()
                first = first or now
                last = now
    out[i] = (n, first, last)


for s in STREAMS:
    out = {}
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i, out)) for i in range(s)]
    [t.start() for t in th]
    [t.join() for t in th]
    t1 = time.time()
    tot = sum(v[0] for v in out.values())
    per = [(v[0] - 1) / (v[2] - v[1]) for v in out.values() if v[2] and v[2] > v[1]]
    print(f"{s:3d} streams: {tot / (t1 - t0):7.1f} tok/s total (wall), decode per stream {sum(per) / len(per):6.1f} tok/s",
          flush=True)
