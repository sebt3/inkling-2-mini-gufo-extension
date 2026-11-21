#!/usr/bin/env python3
"""Query the oracle llama-server: tokenize a text, then for several prefix lengths fetch the top-N logprobs of the
next token. Writes JSON {tokens:[...], positions:{"<len>": [[id, logprob], ...]}}.
usage: oracle_topk.py BASE_URL TEXT_FILE OUT.json [--lens 1,5,10] [--topk 20]"""
import json, sys, urllib.request

def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def main():
    base, textf, out = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    lens = [int(x) for x in args[args.index("--lens") + 1].split(",")] if "--lens" in args else None
    topk = int(args[args.index("--topk") + 1]) if "--topk" in args else 20
    text = open(textf, encoding="utf-8").read()
    toks = post(base + "/tokenize", {"content": text, "add_special": False})["tokens"]
    if lens is None:
        lens = sorted({1, 2, 4, 8, 16, 32, len(toks)} & set(range(1, len(toks) + 1)))
    res = {}
    for n in lens:
        r = post(base + "/completion", {"prompt": toks[:n], "n_predict": 1, "n_probs": topk, "temperature": 0,
                                         "cache_prompt": False, "post_sampling_probs": False})
        cp = r["completion_probabilities"][0]
        tl = cp.get("top_logprobs") or cp.get("top_probs")
        res[str(n)] = [[e["id"], e.get("logprob", e.get("prob"))] for e in tl]
    json.dump({"tokens": toks, "positions": res}, open(out, "w"))
    print("tokens:", len(toks), "positions:", list(res))
main()
