#!/usr/bin/env python3
"""Compare engine logits (f32, n_vocab each per position) with oracle top-N logprobs (oracle_topk.py output).
usage: compare_topk.py LOGITS.bin ORACLE.json [--vocab 200058]
Gate G3: top-1 agrees at every position where the oracle margin > 0.05, mean |dlogprob| over oracle top-20 < 0.05."""
import json, math, struct, sys
import array
def main():
    lb, oj = sys.argv[1], sys.argv[2]
    V = int(sys.argv[sys.argv.index("--vocab") + 1]) if "--vocab" in sys.argv else 200058
    o = json.load(open(oj)); raw = open(lb, "rb").read(); n = len(raw) // (4 * V)
    bad = 0; tot = 0.0; cnt = 0; worst = 0.0
    for k, top in sorted(o["positions"].items(), key=lambda kv: int(kv[0])):
        pos = int(k) - 1
        if pos >= n: continue
        a = array.array("f"); a.frombytes(raw[pos * V * 4:(pos + 1) * V * 4])
        mx = max(a); z = math.log(sum(math.exp(x - mx) for x in a))
        lp = lambda i: a[i] - mx - z
        top1 = max(range(V), key=lambda i: a[i])
        ok = top1 == top[0][0] or (len(top) > 1 and top[0][1] - top[1][1] < 0.05)
        diffs = [abs(lp(i) - l) for i, l in top]
        md = sum(diffs) / len(diffs); worst = max(worst, max(diffs)); tot += md; cnt += 1
        if not ok: bad += 1
        print(f"len {int(k):3d}: top1 engine={top1} oracle={top[0][0]} {'OK' if ok else 'MISMATCH'}  mean|dlp|={md:.4f} max={max(diffs):.4f}")
    print(f"positions={cnt} mismatches={bad} mean|dlp|={tot/max(cnt,1):.4f} worst={worst:.4f}")
    sys.exit(1 if bad or tot / max(cnt, 1) > 0.05 else 0)
main()
