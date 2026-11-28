"""Stability run for the two-Spark server: for MINUTES, keep STREAMS concurrent requests going, mixing short questions,
300-token essays and long prompts (8K and 32K tokens of filler + a question, which take the grouped-expert path). Logs
every request (prompt/completion tokens, seconds, tok/s, finish) and fails on any HTTP error, empty completion or a long
prompt whose answer misses the planted fact.  Usage: soak.py URL MINUTES STREAMS"""
import json, random, sys, threading, time, urllib.request
url, minutes, streams = sys.argv[1] + "/v1/chat/completions", float(sys.argv[2]), int(sys.argv[3])
FILL = "The quick survey of regional weather stations recorded routine readings with no unusual events that day. "
def long_prompt(n_tok):
    k = n_tok // 18; pos = random.randrange(k)
    parts = [FILL] * k; parts[pos] = "Note for the reader: the access code for the north gate is 7316. "
    return "".join(parts) + "\n\nWhat is the access code for the north gate? Answer with the number only.", "7316"
SHORT = ["What is the boiling point of water at sea level in Fahrenheit?", "Name three moons of Jupiter.",
         "Write a haiku about snow on mountains.", "What does DNA stand for?"]
ESSAY = "Explain how a refrigerator moves heat from inside to outside, step by step."
stop = time.time() + minutes * 60; lock = threading.Lock(); fails = []; done = []
def one(kind):
    if kind == "long8k": p, want = long_prompt(8000); mt = 200
    elif kind == "long32k": p, want = long_prompt(32000); mt = 200
    elif kind == "essay": p, want, mt = ESSAY, None, 300
    else: p, want, mt = random.choice(SHORT), None, 200
    body = {"model": "Inkling-2-Small", "max_tokens": mt, "temperature": 0, "messages": [{"role": "user", "content": p}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
            j = json.load(r)
    except Exception as e:
        return {"kind": kind, "error": repr(e)[:200]}
    m = j["choices"][0]["message"]; txt = (m.get("content") or "") + (m.get("reasoning_content") or m.get("reasoning") or "")
    u = j["usage"]; dt = time.time() - t
    row = {"kind": kind, "prompt": u["prompt_tokens"], "completion": u["completion_tokens"], "s": round(dt, 1),
           "finish": j["choices"][0]["finish_reason"], "ok": bool(txt.strip()) and (want is None or want in txt)}
    if not row["ok"]: row["text"] = txt[-300:]
    return row
def worker(i):
    kinds = ["short", "essay", "short", "long8k", "essay", "short", "long32k"]
    n = i
    while time.time() < stop:
        row = one(kinds[n % len(kinds)]); n += 1
        with lock:
            done.append(row); print(time.strftime("[%H:%M:%S]"), json.dumps(row), flush=True)
            if "error" in row or not row["ok"]: fails.append(row)
ts = [threading.Thread(target=worker, args=(i,)) for i in range(streams)]; [t.start() for t in ts]; [t.join() for t in ts]
ok = [r for r in done if "error" not in r]
print(f"SUMMARY {len(done)} requests, {len(fails)} failed; completion tokens {sum(r['completion'] for r in ok)}; "
      f"long prompts right {sum(r['ok'] for r in ok if r['kind'].startswith('long'))}/{sum(1 for r in ok if r['kind'].startswith('long'))}", flush=True)
sys.exit(1 if fails else 0)
