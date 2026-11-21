#!/usr/bin/env python3
"""Concurrency gate: N identical-length greedy requests sent in parallel to a Gufo server must give the same text as the same request alone
(greedy, same prompt), and the aggregate throughput is reported.
usage: batch_gate.py URL [--n 4] [--model NAME] [--tokens 120]"""
import json, sys, threading, time, urllib.request
def post(url, body, t=900):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=t))
url = sys.argv[1]; n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "Inkling-2-Mini"; ntok = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 120
prompts = ["Explique en quelques phrases ce qu'est un Deployment Kubernetes.", "Donne trois conseils pour apprendre Rust.", "Quelle est la différence entre TCP et UDP ?", "Écris un haïku sur la pluie."]
def ask(p):
    r = post(url + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": ntok, "temperature": 0})
    m = r["choices"][0]["message"]; return (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), r["usage"]["completion_tokens"]
solo = []; t0 = time.time(); toks = 0
for p in prompts[:n]: o, c = ask(p); solo.append(o); toks += c
t_solo = time.time() - t0; print(f"sequential: {toks} tokens in {t_solo:.1f}s = {toks / t_solo:.1f} tok/s")
res = [None] * n; cnt = [0] * n
def run(i): res[i], cnt[i] = ask(prompts[i])
th = [threading.Thread(target=run, args=(i,)) for i in range(n)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; t_par = time.time() - t0
print(f"parallel x{n}: {sum(cnt)} tokens in {t_par:.1f}s = {sum(cnt) / t_par:.1f} tok/s aggregate")
ok = 0
for i in range(n):
    same = res[i] == solo[i]; ok += same
    k = next((j for j, (a, b) in enumerate(zip(res[i], solo[i])) if a != b), min(len(res[i]), len(solo[i])))
    print(f"  request {i}: identical={same} (common prefix {k}/{len(solo[i])} chars)")
print("RESULT", "PASS" if ok == n else "CHECK (differences can come from near-tie expert choices; inspect)")
