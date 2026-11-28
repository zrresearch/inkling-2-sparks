"""Single-stream decode speed of the two-Spark server: one fixed prompt, 300 tokens, streamed; prints prompt and decode
rates. Usage: speedcheck.py [URL] [label]"""
import json, sys, time, urllib.request
url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8899") + "/v1/chat/completions"
body = {"model": "Inkling-2-Small", "max_tokens": 300, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Explain how a refrigerator moves heat from inside to outside, step by step, "
                      "including the role of the compressor, condenser, expansion valve and evaporator."}],
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); first = None; n = 0; usage = None
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            dl = d["choices"][0]["delta"] if d.get("choices") else {}
            if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                n += 1
                if first is None: first = time.time()
t1 = time.time()
k = usage["completion_tokens"] if usage else n
print(f"{sys.argv[2] if len(sys.argv) > 2 else ''}: prompt {usage['prompt_tokens'] if usage else '?'} tok, first token {first - t0:.1f}s, "
      f"{k - 1} more tokens in {t1 - first:.1f}s = {(k - 1) / (t1 - first):.2f} tok/s")
