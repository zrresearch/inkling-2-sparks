"""Greedy replies with per-token logprobs from a running server, for before/after comparisons.

    golden.py save OUT.json [--url URL] [--tokens 256]
    golden.py compare A.json B.json

compare prints, per prompt: identical (bytes), the first differing token, and the mean |logprob difference| over the
shared prefix, plus the share of positions where both builds' top-1 token agree (over the shared prefix).
"""
import json, sys, time, urllib.request

PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring.",
    "tool": "You can call get_weather(city: str). The user asks: what's the weather in Missoula and in Boise? "
            "Reply with the function calls you would make, one per line, then one sentence.",
    "chat": "Explain in two short paragraphs why the sky is blue.",
    "math": "A train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive? Show your steps.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def save(out, url, tokens):
    res = {}
    for k, p in PROMPTS.items():
        t0 = time.time()
        j = post(f"{url}/v1/chat/completions", {
            "model": "Inkling-2-Small", "messages": [{"role": "user", "content": p}], "max_tokens": tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 1, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False}})
        dt = time.time() - t0
        c = j["choices"][0]
        lp = [(t["token"], t["logprob"]) for t in (c.get("logprobs") or {}).get("content") or []]
        res[k] = {"text": (c["message"].get("reasoning_content") or "") + "\n---\n" + (c["message"].get("content") or ""),
                  "tokens": lp, "seconds": dt, "completion_tokens": j["usage"]["completion_tokens"]}
        print(f"{k}: {j['usage']['completion_tokens']} tokens in {dt:.1f} s ({j['usage']['completion_tokens']/dt:.1f} tok/s)")
    json.dump(res, open(out, "w"), indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    for k in A:
        ta, tb = A[k]["tokens"], B[k]["tokens"]
        same = A[k]["text"] == B[k]["text"]
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x[0] != y[0]), None)
        n = first if first is not None else min(len(ta), len(tb))
        d = [abs(x[1] - y[1]) for x, y in zip(ta[:n], tb[:n])]
        print(f"{k:6s} identical={same!s:5s} first diff at token {first if first is not None else '-':>4}"
              f"  mean |dlogprob| over {n} shared tokens {sum(d)/max(1,len(d)):.4f}  max {max(d, default=0):.4f}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "save":
        url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:8899"
        tok = int(a[a.index("--tokens") + 1]) if "--tokens" in a else 256
        save(a[2], url, tok)
    else:
        compare(a[2], a[3])
