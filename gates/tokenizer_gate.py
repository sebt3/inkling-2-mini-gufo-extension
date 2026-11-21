#!/usr/bin/env python3
"""GATE G1: tokenizer of host/tokenizer.cpp versus the oracle llama-server (/tokenize, /detokenize).

usage:
  tokenizer_gate.py gen > cases.jsonl                        # deterministic corpus, one case per line
  tokenizer_gate.py check --oracle http://IP:8001 --cases cases.jsonl --probe results.jsonl

The probe results come from tools/tokenizer_probe (stdin = cases.jsonl, one {"ids", "text"} line per case).
A case passes when:
  * the ids are identical to the oracle /tokenize (add_special=false, parse_special as in the case), and
  * decode(ids) of the probe gives back the text, and
  * the oracle /detokenize of its own ids gives back the text (checks the round trip on the oracle side too).
Exit status 0 only when every case passes.
"""
import argparse
import json
import random
import sys
import urllib.request

SPECIALS = [
    "<|message_user|>", "<|message_system|>", "<|message_model|>", "<|message_tool|>",
    "<|content_text|>", "<|content_thinking|>", "<|content_image|>", "<|content_invoke_tool_json|>",
    "<|end_message|>", "<|content_model_end_sampling|>", "<|unused_200054|>",
]

HAND = [
    "", " ", "  ", "\n", "\n\n\n", "\t", " \n ", "\r\n", "a", "A", "1", "0123456789",
    "Bonjour, ça va ? Très bien, merci !", "Où est l'élève qui préfère le café crème à l'été ?",
    "Ça coûte 3,50 € à Noël.", "L'œuvre de Léa, Åsa et Øyvind : naïveté, façade, garçon.",
    "The quick brown fox jumps over the lazy dog.", "I'm sure we'll see you've been, they'd say.",
    "fn main() {\n    println!(\"Hello, {}!\", name);\n}\n",
    "def f(x: int) -> int:\n    return x * 2  # double it\n",
    "let r = rhai::Engine::new(); r.eval::<i64>(\"1 + 2\")?;",
    "apiVersion: vinyl.solidite.fr/v1beta1\nkind: Package\nmetadata:\n  name: test\n",
    "{\"name\": \"x\", \"values\": [1, 2.5, -3e10, null, true], \"nested\": {\"k\": \"v\\n\"}}",
    "SELECT id, name FROM users WHERE created_at >= '2026-01-01' ORDER BY id DESC;",
    "$ curl -s https://example.org/api?x=1&y=2 | jq '.data[0]'",
    "a\n\nb\n\n\nc\n   \nd", "multiple     spaces    inside", "  leading and trailing  ",
    "tab\tseparated\tvalues", "CamelCaseIdentifierHTTPServerError", "snake_case_name_2026",
    "ALL CAPS SENTENCE HERE", "mixed UPPER and lower Case WORDS", "x=1; y=2; z=x+y",
    "😀", "😀😀😀", "👍🏽 skin tone", "👨‍👩‍👧 family ZWJ", "🇫🇷 drapeau", "emoji at end 🤖",
    "中文字符测试", "日本語のテキスト、ひらがなとカタカナ", "한국어 텍스트", "中英混合 mixed text 混合",
    "Русский текст", "Ελληνικά κείμενο", "النص العربي", "עברית",
    "é combining acute", "café vs café", "ño", " nbsp here",
    "ideographic　space", "line separator", "zero​width", "\u0085nel",
    "−minus sign 5 ＋ fullwidth", "½ ¼ ② ⅻ numerals", "Ⅻ roman", "٣٤٥ arabic digits",
    "é" * 40, "word " * 60, "x" * 500,
    "<|message_user|>Bonjour<|end_message|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
    "<|message_model|><|content_thinking|>je pense<|end_message|><|message_model|><|content_text|>Oui",
    "texte avant <|content_model_end_sampling|> après",
    "<|message_user|>", "<|message_user|> <|message_user|>", "<|end_message|><|end_message|>",
    "<|message_unknown|> not special <|content_text", "<|message_user", "message_user|>",
    "<|content_image|><|unused_200054|><|end_message|>",
    "line1\n<|message_user|>\nline2 <|content_text|>\n",
    "quoted \"<|message_user|>\" and 'single'", "back\\slash \\n literal",
    "control char \x01 and \x1f here", "del \x7f char", "percent %s %d {} {0}",
    "\\u0041 not an escape", "tab\\t literal", "null\x00byte",
]

WORDS = ["bonjour", "réseau", "système", "café", "naïve", "Kubernetes", "cluster", "pod", "Strix", "Halo",
         "modèle", "fenêtre", "été", "Ünïcödé", "rust", "vector<int>", "fn", "let", "x", "42", "3.14",
         "🙂", "🚀", "中文", "漢字", "かな", "한국", "Привет", "αβγ", "عربي", "→", "…", "“quoted”", "«fr»",
         "_", "__init__", "::", "->", "=>", "&&", "||", "//", "/*", "*/", "#", "$HOME", "~/x", "a.b.c",
         "http://x.y/z?q=1", "--flag", "-v", "\t", "\n", "  ", "   ", "\n\n", "\r\n", "", "0", "1e-5"]


def random_text(rng):
    parts = []
    for _ in range(rng.randint(1, 14)):
        r = rng.random()
        if r < 0.10:
            parts.append(rng.choice(SPECIALS))
        elif r < 0.35:
            parts.append(rng.choice(WORDS))
        elif r < 0.45:
            parts.append(rng.choice([" ", "  ", "\n", "\n\n", "\t", " \n", "\r\n", "\n   "]))
        elif r < 0.55:
            parts.append("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 9))))
        elif r < 0.65:
            parts.append("".join(chr(rng.randint(0x21, 0x7e)) for _ in range(rng.randint(1, 8))))
        elif r < 0.75:
            parts.append("".join(chr(rng.randint(0x4e00, 0x9fff)) for _ in range(rng.randint(1, 4))))
        elif r < 0.85:
            parts.append("".join(chr(rng.randint(0x00e0, 0x017f)) for _ in range(rng.randint(1, 6))))
        elif r < 0.92:
            parts.append(chr(rng.choice([0x1F600, 0x1F680, 0x1F44D, 0x2764, 0x1F1EB, 0x1F469])))
        else:
            parts.append(rng.choice(["", " ", "’", "'", "\"", "é", "ß", "ǅ", "ⅷ", "́"]))
    return "".join(parts)


def corpus():
    cases = [{"text": t, "parse_special": True} for t in HAND]
    cases += [{"text": t, "parse_special": False} for t in HAND]
    rng = random.Random(20261121)
    while len(cases) < 340:
        cases.append({"text": random_text(rng), "parse_special": rng.random() < 0.5})
    return cases


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def check(args):
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    if len(cases) != len(probes):
        print("count mismatch: %d cases, %d probe results" % (len(cases), len(probes)))
        return 1
    passed = 0
    fails = []
    for i, (c, p) in enumerate(zip(cases, probes)):
        oracle_ids = post(args.oracle, "/tokenize",
                          {"content": c["text"], "add_special": False, "parse_special": c["parse_special"]})["tokens"]
        oracle_text = post(args.oracle, "/detokenize", {"tokens": oracle_ids})["content"]
        ids_ok = p["ids"] == oracle_ids
        text_ok = p["text"] == c["text"] and oracle_text == c["text"]
        if ids_ok and text_ok:
            passed += 1
        else:
            fails.append((i, c, oracle_ids, p["ids"], p["text"], oracle_text))
    print("G1 tokenizer: %d/%d cases identical to the oracle (ids and decode round trip)" % (passed, len(cases)))
    for i, c, oids, pids, ptext, otext in fails[:10]:
        print("--- case %d parse_special=%s text=%s" % (i, c["parse_special"], json.dumps(c["text"], ensure_ascii=False)))
        if oids != pids:
            k = next((j for j, (a, b) in enumerate(zip(oids, pids)) if a != b), min(len(oids), len(pids)))
            print("  ids differ at %d: oracle %s | probe %s" % (k, oids[k:k + 8], pids[k:k + 8]))
        if ptext != c["text"]:
            print("  probe decode: %s" % json.dumps(ptext, ensure_ascii=False))
        if otext != c["text"]:
            print("  oracle decode: %s" % json.dumps(otext, ensure_ascii=False))
    return 0 if passed == len(cases) else 1


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
