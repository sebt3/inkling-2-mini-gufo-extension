#!/usr/bin/env python3
"""GATE G2: prompt rendering of host/chat_template.cpp versus the oracle llama-server (/apply-template).

usage:
  template_gate.py gen > cases.jsonl                       # deterministic corpus, one case per line
  template_gate.py check --oracle http://IP:8001 --cases cases.jsonl --probe results.jsonl

The probe results come from tools/template_probe (stdin = cases.jsonl). A case passes when the oracle and the probe
return the same prompt, or both reject the conversation. Exit status 0 only when every case passes.
Cases with prefill_assistant=false are not checked against the oracle (the server always prefills).
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request

EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def msg(role, content="__absent__", reasoning=None, name=None):
    m = {"role": role}
    if content != "__absent__":
        m["content"] = content
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if name is not None:
        m["name"] = name
    return m


def case(messages, effort=None, value=None, prefill=True):
    return {"messages": messages, "reasoning_effort": effort, "reasoning_effort_value": value,
            "prefill_assistant": prefill}


def corpus():
    """Hand-written conversations first, then seeded random ones."""
    S = msg("system", "You are a helpful assistant.")
    U = lambda t: msg("user", t)
    A = lambda t, r=None, name=None: msg("assistant", t, reasoning=r, name=name)
    cases = [
        # system only / no system / empty
        [S],
        [],
        [U("hi")],
        [S, U("hi")],
        [S, U("Bonjour, ça va ?")],
        [msg("system", "")],
        [msg("system", "s1"), msg("system", "s2"), U("x")],
        # multi-turn, closed assistant turns
        [S, U("a"), A("b"), U("c")],
        [U("a"), A("b"), U("c"), A("d"), U("e")],
        [S, U("q1"), A("r1"), U("q2"), A("r2"), U("q3")],
        [U("a"), A("b"), msg("system", "late system"), U("c")],
        [U("a"), A("b"), msg("tool", "result", name="get_weather"), U("c")],
        [U("a"), A("b"), msg("tool", "result")],
        [U("a"), A("b", r="thinking first"), U("c")],
        [U("a"), A("", r="only thinking"), U("c")],
        [U("a"), A("", r=None), U("c")],
        [U("a"), msg("assistant", None), U("c")],
        [U("a"), msg("assistant", None, reasoning="think"), U("c")],
        [U("a"), msg("assistant", "answer", name="bot"), U("c")],
        [U("a"), A("b", r="")],
        # empty and odd contents
        [U("")],
        [U(" ")],
        [U("\n\n\n")],
        [U("   spaces   everywhere   ")],
        [msg("user", None)],
        [msg("user", [])],
        [msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])],
        [msg("user", [{"type": "text", "text": ""}])],
        [msg("user", [{"type": "text", "text": "one"}]), A("two")],
        # special characters and markers inside content
        [U("line1\nline2\r\n\tTabbed \"quoted\" \\ backslash")],
        [U("<|message_user|> <|end_message|> <|content_text|> inside content")],
        [U("ü é 中文 😀 — « guillemets » œuvre ÇA")],
        [U("code:\n```rust\nfn main() { println!(\"hi\"); }\n```")],
        [U("{\"json\": [1, 2, {\"k\": null}]}")],
        [A("émoji 🤖 et CJK 日本語 suivi")],
        # assistant at the end: prefilled continuation (server default)
        [S, U("Raconte-moi une histoire."), A("Il était une fois")],
        [S, U("q"), A("Il était une fois, ")],
        [U("q"), A("Réponse partielle\n")],
        [U("q"), A("   trailing spaces   ")],
        [U("q"), A("", r="Je réfléchis encore")],
        [U("q"), A("Oui", r="Je dois d'abord réfléchir")],
        [U("q"), A("Oui", r="Pensée avec\nsauts de ligne")],
        [U("q"), msg("assistant", "")],
        [U("q"), msg("assistant", [])],
        [U("q"), msg("assistant", [{"type": "text", "text": "partie 1"}, {"type": "text", "text": "partie 2"}])],
        [U("q"), msg("assistant", None, reasoning="pensée seule")],
        [msg("user")],
        [U("a"), msg("user"), U("c")],
        [U("q"), A("une réponse", r="raison")],
        [msg("assistant", "seul en premier")],
        [msg("assistant", "", reasoning="seulement pensée"), U("et ensuite ?")],
        [S, U("q"), A("x"), U("y"), A("z")],
        [U("q"), A("x"), U("y"), A("z", r="raison finale")],
        [U("q"), A("réponse", name="nom")],
        # efforts (template default is 0.9)
        [U("a")],
    ]
    efforts = [
        (None, None), ("high", None), ("low", None), ("minimal", None), ("medium", None), ("xhigh", None),
        ("max", None), ("none", None), (" high ", None), ("bogus", None), (None, 0.0), (None, 0.5),
        (None, 0.25), (None, 0.123456789), (None, 0.99), (None, 0.00001), (None, 0.7), (None, 1.0),
        (None, -0.1),
    ]
    out = []
    for i, conv in enumerate(cases):
        eff, val = efforts[i % len(efforts)]
        out.append(case(conv, eff, val))
    # the same conversations with explicit efforts
    for i, conv in enumerate(cases[:12]):
        out.append(case(conv, EFFORT_LEVELS[i % len(EFFORT_LEVELS)]))
    # errors (both sides must reject)
    out.append(case([msg("alien", "x")]))
    out.append(case([U("a"), msg("assistant")]))
    out.append(case([U("a"), msg("assistant"), U("b")]))
    out.append(case([U("a"), A("b"), A("c")]))
    out.append(case([msg("user", [{"type": "image_url", "image_url": {"url": "x"}}])]))
    out.append(case([U("a"), A("b", r="c"), msg("assistant")]))
    # seeded random conversations
    rng = random.Random(20261121)
    words = ["ok", "Bonjour", "Rust", "Ünïcödé", "a\nb", "  ", "1234", "😀", "中文", "<|x|>", "\"q\"", "\\"]
    for _ in range(40):
        n = rng.randint(1, 6)
        conv = []
        if rng.random() < 0.5:
            conv.append(msg("system", " ".join(rng.choice(words) for _ in range(rng.randint(0, 4)))))
        for k in range(n):
            role = "user" if k % 2 == 0 else "assistant"
            text = " ".join(rng.choice(words) for _ in range(rng.randint(0, 5)))
            r = rng.choice([None, None, "pense", ""]) if role == "assistant" else None
            conv.append(msg(role, text, reasoning=r))
        eff = rng.choice([None, None, "high", "low", "max"])
        val = rng.choice([None, None, None, round(rng.random() * 0.99, 3)]) if eff is None else None
        out.append(case(conv, eff, val))
    return out


def oracle_prompt(base, c):
    body = {"messages": c["messages"]}
    if c["reasoning_effort"] is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": c["reasoning_effort"]}
    if c["reasoning_effort_value"] is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": c["reasoning_effort_value"]}
    req = urllib.request.Request(base + "/apply-template", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))["prompt"], None
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:200]


def check(args):
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    if len(cases) != len(probes):
        print("count mismatch: %d cases, %d probe results" % (len(cases), len(probes)))
        return 1
    checked = passed = 0
    fails = []
    for i, (c, p) in enumerate(zip(cases, probes)):
        if not c["prefill_assistant"]:
            continue
        checked += 1
        oprompt, oerr = oracle_prompt(args.oracle, c)
        pprompt, perr = p.get("prompt"), p.get("error")
        if oprompt is None and perr is not None:
            passed += 1
        elif oprompt is not None and pprompt == oprompt:
            passed += 1
        else:
            fails.append((i, c, oprompt, oerr, pprompt, perr))
    print("G2 template: %d/%d identical to the oracle (%d cases with prefill=false not checked)"
          % (passed, checked, len(cases) - checked))
    for i, c, oprompt, oerr, pprompt, perr in fails[:10]:
        print("--- case %d: %s" % (i, json.dumps(c, ensure_ascii=False)))
        print("  oracle: %s" % (repr(oprompt) if oprompt is not None else "ERROR " + str(oerr)))
        print("  probe : %s" % (repr(pprompt) if pprompt is not None else "ERROR " + str(perr)))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen")
    c = sub.add_parser("check")
    c.add_argument("--oracle", required=True)
    c.add_argument("--cases", required=True)
    c.add_argument("--probe", required=True)
    args = ap.parse_args()
    if args.cmd == "gen":
        for i, cs in enumerate(corpus()):
            print(json.dumps(cs, ensure_ascii=False))
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
