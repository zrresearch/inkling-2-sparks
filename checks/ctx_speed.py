"""Prompt-reading speed and decode speed at long context, one stream.

For each prompt length N: a prompt of N tokens (WikiText repeated, with a unique nonce first so no prefix-cache hit
helps), sent once with max_tokens=1 (prompt seconds -> prompt tok/s), then once more (new nonce) streaming 256 tokens
(decode tok/s at that context, from first to last token).
  python3 ctx_speed.py [URL] [--lengths 4096,32768,131072]
"""
import json, sys, time, uuid, urllib.request

URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8899")
LENS = [4096, 32768, 131072]
if "--lengths" in sys.argv:
    LENS = [int(x) for x in sys.argv[sys.argv.index("--lengths") + 1].split(",")]
MODEL = "Inkling-2-Small"


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


base = post("/tokenize", {"model": MODEL, "prompt": open("wiki16k.txt").read()}).read()
base = json.loads(base)["tokens"]


def prompt(n):
    nonce = json.loads(post("/tokenize", {"model": MODEL, "prompt": f"[{uuid.uuid4()}]\n"}).read())["tokens"]
    body = (base * (n // len(base) + 1))[: n - len(nonce)]
    return nonce + body


print(f"{'tokens':>8} {'prompt s':>9} {'prompt tok/s':>13} {'decode tok/s':>13}", flush=True)
for n in LENS:
    t0 = time.time()
    json.loads(post("/v1/completions", {"model": MODEL, "prompt": prompt(n), "max_tokens": 1, "temperature": 0}).read())
    ps = time.time() - t0
    r = post("/v1/completions", {"model": MODEL, "prompt": prompt(n), "max_tokens": 256, "temperature": 0,
                                 "stream": True, "ignore_eos": True})
    first = last = None
    count = 0
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        if json.loads(line[5:])["choices"][0].get("text"):
            last = time.time()
            first = first or last
            count += 1
    dec = (count - 1) / (last - first) if count > 1 else float("nan")
    print(f"{n:>8} {ps:>9.1f} {n / ps:>13.0f} {dec:>13.1f}", flush=True)
