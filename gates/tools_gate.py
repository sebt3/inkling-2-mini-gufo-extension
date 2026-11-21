#!/usr/bin/env python3
"""GATE G9a: prompt rendering with tools (host/chat_template.cpp) versus the oracle llama-server (/apply-template).

usage:
  tools_gate.py gen > cases.jsonl                          # deterministic corpus, one case per line
  tools_gate.py check --oracle http://IP:8001 --cases cases.jsonl --probe results.jsonl

The probe results come from tools/tools_probe (stdin = cases.jsonl). A case passes when the oracle and the probe
return the same prompt string, or both reject the conversation. Exit status 0 only when every case passes.
Cases with prefill_assistant=false are not checked (the server always prefills).
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request

EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def msg(role, content="__absent__", reasoning=None, name=None, tool_calls=None, tool_call_id=None):
    m = {"role": role}
    if content != "__absent__":
        m["content"] = content
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if name is not None:
        m["name"] = name
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    if tool_call_id is not None:
        m["tool_call_id"] = tool_call_id
    return m


def call(name, args, cid=None, typ="function"):
    c = {"type": typ, "function": {"name": name, "arguments": args}}
    if cid is not None:
        c["id"] = cid
    return c


def tool(name, description=None, parameters="__absent__", typ="function", wrap=True):
    fn = {"name": name}
    if description is not None:
        fn["description"] = description
    if parameters != "__absent__":
        fn["parameters"] = parameters
    if not wrap:
        return fn
    t = {"type": typ, "function": fn}
    return t


def case(messages, tools=None, effort=None, value=None, prefill=True):
    return {"messages": messages, "tools": tools, "reasoning_effort": effort,
            "reasoning_effort_value": value, "prefill_assistant": prefill}


WEATHER = tool("get_weather", "Get the weather for a city.", {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    "required": ["city"],
})
TIME = tool("get_time", "Current time in a timezone", {"type": "object", "properties": {"tz": {"type": "string"}}})
NESTED = tool("book_trip", "Book a trip; accents é, quotes \" and slashes / are fine.", {
    "type": "object",
    "properties": {
        "legs": {"type": "array", "items": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}, "stops": {"type": "integer", "minimum": 0}}}},
        "budget": {"type": "number", "maximum": 1e20, "minimum": 0.5, "default": 1.0},
        "refundable": {"type": "boolean"},
        "note": {"type": "null"},
    },
    "required": ["legs"],
})
NOPARAMS = tool("ping")
EMPTY_DESC = tool("empty", "", {})

U = lambda t: msg("user", t)
A = lambda t=None, **kw: msg("assistant", t, **kw)
TOOLMSG = lambda t, **kw: msg("tool", t, **kw)


def hand_written():
    cs = []
    # one tool, one or several, declarations with nested parameters
    cs.append(case([U("What's the weather in Paris?")], [WEATHER]))
    cs.append(case([U("time?")], [WEATHER, TIME]))
    cs.append(case([msg("system", "You are helpful."), U("book it")], [NESTED]))
    cs.append(case([U("hi")], [NOPARAMS, EMPTY_DESC]))
    cs.append(case([U("hi")], [], effort="high"))
    cs.append(case([U("hi")], None))
    cs.append(case([U("hi")], [WEATHER], effort="low"))
    cs.append(case([U("hi")], [WEATHER], value=0.25))
    cs.append(case([U("hi")], [tool("bare", wrap=False)]))  # no "function" key: must be rejected on both sides
    # tool call by an assistant without content, with content, with reasoning, parts
    cs.append(case([U("weather?"), A(tool_calls=[call("get_weather", {"city": "Paris"}, "c1")]), U("thanks")],
                   [WEATHER]))
    cs.append(case([U("weather?"), A("Let me check.", tool_calls=[call("get_weather", {"city": "Paris", "unit": "c"})]),
                    U("next")], [WEATHER]))
    cs.append(case([U("q"), A("", reasoning="I should call the tool", tool_calls=[call("get_time", {})]), U("ok")],
                   [TIME]))
    cs.append(case([U("q"), A("done", reasoning="thinking first", tool_calls=[call("get_time", {"tz": "UTC"})]),
                    TOOLMSG("12:00", name="get_time"), A("final answer", reasoning="second thoughts"), U("x")],
                   [TIME]))
    cs.append(case([U("q"), msg("assistant", [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}],
                                tool_calls=[call("get_time", {"tz": "CET"})]), U("y")], [TIME]))
    cs.append(case([U("q"), msg("assistant", None, tool_calls=[call("get_time", {"tz": "CET"})]), U("y")], [TIME]))
    cs.append(case([U("q"), A("", tool_calls=[call("get_time", {"tz": "x"})]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[]), U("y")], [TIME]))  # empty call list: no call rendered
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {}), call("get_weather", {"city": "Rome"})]), U("y")],
                   [TIME, WEATHER]))
    cs.append(case([U("q"), A("a", tool_calls=[call("get_time", {"tz": "x"}, "t1")], name="get_time"), U("y")],
                   [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})], name="other"), U("y")], [TIME]))  # name mismatch
    # tool messages: by name, by tool_call_id, both, neither, unknown id, duplicated id
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {"tz": "a"}, "k1"), call("get_weather", {"city": "B"}, "k2")]),
                    TOOLMSG("r2", tool_call_id="k2"), TOOLMSG("r1", tool_call_id="k1"), U("z")], [TIME, WEATHER]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {}, "k1")]), TOOLMSG("r", name="explicit", tool_call_id="k1"),
                    U("z")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {}, "k1")]), TOOLMSG("r", tool_call_id="nope"), U("z")],
                   [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), TOOLMSG("r"), U("z")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {}, "dup"), call("get_weather", {"city": "x"}, "dup")]),
                    TOOLMSG("r", tool_call_id="dup"), U("z")], [TIME, WEATHER]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), TOOLMSG("", tool_call_id="x"), U("z")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]),
                    TOOLMSG([{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}], name="get_time"),
                    U("z")], [TIME]))
    # arguments: nested, unicode, sorting, string JSON, empty, null, falsy scalars
    cs.append(case([U("q"), A(tool_calls=[call("book_trip", {"legs": [{"from": "Lyon", "to": "Nice", "stops": 2}],
                                                             "budget": 1234.5, "refundable": False, "note": None})]),
                    U("y")], [NESTED]))
    cs.append(case([U("q"), A(tool_calls=[call("get_weather", {"zeta": 1, "Alpha": 2, "été": "é\"\n\t\\/ <>&", "b": {"y": 2, "x": 1}})]),
                    U("y")], [WEATHER]))
    cs.append(case([U("q"), A(tool_calls=[call("get_weather", '{"city": "Paris", "unit": "c"}')]), U("y")], [WEATHER]))
    cs.append(case([U("q"), A(tool_calls=[call("get_weather", "{\"b\": 1, \"a\": [1.0, 2.5e-7, 1e20, -0.0, 100]}")]), U("y")], [WEATHER]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", None)]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", False)]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", 0)]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", "[]")]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", "\"x\"")]), U("y")], [TIME]))  # string after parse: error
    cs.append(case([U("q"), A(tool_calls=[call("get_time", "not json")]), U("y")], [TIME]))  # parse error
    cs.append(case([U("q"), A(tool_calls=[call("get_time", [1, 2])]), U("y")], [TIME]))  # array: error
    cs.append(case([U("q"), A(tool_calls=[call("get_time", 3.5)]), U("y")], [TIME]))  # scalar: error
    cs.append(case([U("q"), A(tool_calls=[{"type": "function", "function": {"name": "get_time"}}]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("a\"b/é<>&", {"we ird\"key": "v\\"}, "c9")]), U("y")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {"n": 1.5, "e": 1e30, "tiny": 0.00001, "big": 1e15})]), U("y")],
                   [TIME]))
    # prefill: a trailing assistant message with tool calls is an error; a trailing text is a continuation
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})])], [TIME]))
    cs.append(case([U("q"), A("Let me see", tool_calls=[call("get_time", {})])], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), A("two")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), TOOLMSG("12:00", name="get_time"), A("It is ")],
                   [TIME]))
    cs.append(case([U("q"), A("answer", reasoning="why")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})])], [TIME], prefill=False))
    # declarations: tool and function fields, types, errors
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": None}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": False, "description": "d"}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": [1, "x"]}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": "text"}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "description": 0}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "description": None}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"description": "no name"}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": 1}}]))
    cs.append(case([U("q")], [{"function": {"name": "t"}}]))
    cs.append(case([U("q")], [{"type": "code_interpreter", "function": {"name": "t"}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "defer_loading": "yes"}}]))
    cs.append(case([U("q")], [{"type": "function", "defer_loading": True, "function": {"name": "t"}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "aé\n\t\"\\/", "parameters": {"Z": {}, "a": {}, "é": {}}}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": {"z": 1, "a": {"y": 2, "b": [3, {"d": 1, "c": 2}]}}}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": {"nums": [0.5, 1.0, 1e-7, 3.14159265358979, 123456789012345678, -0]}}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": {"s": "\u0001\u001f\u007f  😀 é"}}}]))
    cs.append(case([U("q")], [{"type": "function", "function": {"name": "t", "parameters": {"x": 0.0001, "y": 0.00001, "z": 1e16, "w": 1e15}}}]))
    # message-level errors (both sides must reject)
    cs.append(case([U("q"), msg("assistant")], [TIME]))
    cs.append(case([msg("alien", "x")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), A(tool_calls=[call("get_time", {})])], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {})]), msg("tool", None, tool_call_id="k1")], [TIME]))
    cs.append(case([U("q"), A(tool_calls=[call("get_time", {}, "k1")]), TOOLMSG("r", tool_call_id="k1"), U("z")],
                   [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]))
    return cs


def random_cases(n):
    rng = random.Random(20261121)
    words = ["ok", "Bonjour", "Rust", "Ünïcödé", "a\nb", "  ", "1234", "😀", "中文", "<|x|>", "\"q\"", "\\", "é"]
    names = ["get_time", "get_weather", "search", "calc"]
    out = []
    for _ in range(n):
        tools = []
        for nm in rng.sample(names, rng.randint(1, 3)):
            params = {"type": "object", "properties": {
                rng.choice(["q", "é", "a", "Z"]): {"type": rng.choice(["string", "number", "integer"]),
                                                     "description": " ".join(rng.choice(words) for _ in range(2))}}}
            tools.append(tool(nm, rng.choice([None, "desc " + rng.choice(words)]), params))
        conv = [U(" ".join(rng.choice(words) for _ in range(rng.randint(1, 4))))]
        pending = {}
        for _ in range(rng.randint(1, 3)):
            calls = []
            for k in range(rng.randint(1, 2)):
                cname = rng.choice(names)
                args = {rng.choice(["q", "é", "a", "Z"]): rng.choice([1, 2.5, "x é", True, None, [1, "y"], {"k": 0}])}
                cid = "id%d" % rng.randint(0, 999999)
                calls.append(call(cname, args, cid))
                pending[cid] = cname
            text = rng.choice([None, "", " ".join(rng.choice(words) for _ in range(2))])
            reasoning = rng.choice([None, None, "pense"])
            conv.append(msg("assistant", text if text is not None else "", reasoning=reasoning, tool_calls=calls)
                        if text is not None else msg("assistant", None, reasoning=reasoning, tool_calls=calls))
            for c in calls:
                if rng.random() < 0.5:
                    conv.append(TOOLMSG(" ".join(rng.choice(words) for _ in range(2)), name=c["function"]["name"]))
                else:
                    conv.append(TOOLMSG("res", tool_call_id=c["id"]))
            conv.append(U(" ".join(rng.choice(words) for _ in range(rng.randint(0, 3)))))
        effort = rng.choice([None, None, "high", "low", "max"])
        value = None if effort else rng.choice([None, 0.5])
        out.append(case(conv, tools if rng.random() < 0.9 else None, effort, value))
    return out


def corpus():
    return hand_written() + random_cases(60)


def oracle_prompt(base, c):
    body = {"messages": c["messages"]}
    if c["tools"] is not None:
        body["tools"] = c["tools"]
    if c["reasoning_effort"] is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": c["reasoning_effort"]}
    if c["reasoning_effort_value"] is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": c["reasoning_effort_value"]}
    req = urllib.request.Request(base + "/apply-template", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))["prompt"], None
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:300]


def check(args):
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    if len(cases) != len(probes):
        print("count mismatch: %d cases, %d probe results" % (len(cases), len(probes)))
        return 1
    checked = passed = errors = 0
    fails = []
    for i, (c, p) in enumerate(zip(cases, probes)):
        if not c["prefill_assistant"]:
            continue
        checked += 1
        oprompt, oerr = oracle_prompt(args.oracle, c)
        pprompt, perr = p.get("prompt"), p.get("error")
        if oprompt is None and perr is not None:
            passed += 1
            errors += 1
        elif oprompt is not None and pprompt == oprompt:
            passed += 1
        else:
            fails.append((i, c, oprompt, oerr, pprompt, perr))
    print("G9a tools: %d/%d identical to the oracle (%d of them are rejections on both sides; "
          "%d cases with prefill=false not checked)" % (passed, checked, errors, len(cases) - checked))
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
        for cs in corpus():
            print(json.dumps(cs, ensure_ascii=False))
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
