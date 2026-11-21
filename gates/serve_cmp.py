#!/usr/bin/env python3
"""Compare a Gufo server with the oracle llama-server on greedy chat completions.
usage: serve_cmp.py GUFO_URL ORACLE_URL FILE.txt [--model NAME] [--max-tokens N]
Scenarios: multi-turn, long summary of FILE (~its size), needle in a long haystack built from FILE (repeated).
Reports the length of the common prefix of reasoning+content between the two servers and the timings."""
import json, random, sys, time, urllib.request

def post(url, body, t=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))

def ask(base, model, messages, n):
    t0 = time.time()
    r = post(base + "/v1/chat/completions", {"model": model, "messages": messages, "max_tokens": n, "temperature": 0, "stream": False})
    m = r["choices"][0]["message"]; u = r.get("usage", {})
    return (m.get("reasoning_content") or "") + "┃" + (m.get("content") or ""), time.time() - t0, u

def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def main():
    g, o, f = sys.argv[1], sys.argv[2], sys.argv[3]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "Inkling-2-Mini"
    n = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else 300
    doc = open(f, encoding="utf-8").read()
    random.seed(7)
    code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    hay = []
    while sum(len(x) for x in hay) < 90000: hay.append(doc)   # about 25k tokens of repetitive but real text
    mid = len(hay) // 2
    hay.insert(mid, f"\n\nNOTE INTERNE : le code du coffre est {code}.\n\n")
    scenarios = {
        "multi-turn": [{"role": "system", "content": "Tu es un assistant concis."}, {"role": "user", "content": "Donne-moi trois fruits rouges."},
                       {"role": "assistant", "content": "Fraise, framboise, cerise."}, {"role": "user", "content": "Et trois fruits jaunes ?"}],
        "summary": [{"role": "user", "content": "Voici un document technique :\n\n" + doc + "\n\nRésume-le en trois phrases."}],
        "needle": [{"role": "user", "content": "".join(hay) + "\n\nQuel est le code du coffre ? Réponds uniquement par le code."}],
    }
    ok = True
    for name, msgs in scenarios.items():
        res = {}
        for label, base in (("gufo", g), ("oracle", o)):
            try: res[label] = ask(base, model if label == "gufo" else "x", msgs, n)
            except Exception as e: res[label] = ("ERROR " + str(e), 0.0, {})
        a, b = res["gufo"][0], res["oracle"][0]; c = common(a, b)
        u = res["gufo"][2]
        print(f"== {name}: prompt_tokens={u.get('prompt_tokens')} gufo {res['gufo'][1]:.1f}s oracle {res['oracle'][1]:.1f}s; common prefix {c}/{min(len(a), len(b))} chars")
        if name == "needle":
            print("   expected code:", code, "| gufo has it:", code in a, "| oracle has it:", code in b); ok = ok and code in a
        else:
            print("   gufo  :", a[:240].replace("\n", " ")); print("   oracle:", b[:240].replace("\n", " "))
        ok = ok and c >= 0.5 * min(len(a), len(b))
    print("RESULT", "PASS" if ok else "CHECK")
main()
