#!/usr/bin/env python3
"""Describe an image with a Gufo server and the oracle llama-server (greedy), compare the answers.
usage: vision_cmp.py GUFO_URL ORACLE_URL IMAGE.png [--prompt TEXT] [--model NAME]"""
import base64, json, sys, time, urllib.request
def post(url, body, t=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))
g, o, img = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = sys.argv[sys.argv.index("--prompt") + 1] if "--prompt" in sys.argv else "Décris précisément cette image : formes, couleurs et positions."
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "Inkling-2-Mini"
url = "data:image/png;base64," + base64.b64encode(open(img, "rb").read()).decode()
msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}]
for name, base in (("gufo", g), ("oracle", o)):
    t0 = time.time()
    try:
        r = post(base + "/v1/chat/completions", {"model": model if name == "gufo" else "x", "messages": msgs, "max_tokens": 600, "temperature": 0})
        m = r["choices"][0]["message"]; u = r.get("usage", {})
        print(f"== {name} {time.time()-t0:.1f}s prompt_tokens={u.get('prompt_tokens')}")
        print("reasoning:", (m.get("reasoning_content") or "")[:400].replace("\n", " "))
        print("content:", (m.get("content") or "")[:600].replace("\n", " "))
    except Exception as e:
        print("==", name, "ERROR", e)
