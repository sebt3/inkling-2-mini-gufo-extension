#!/usr/bin/env python3
"""Compare two logits dumps (f32, n_vocab per position): top-1 agreement and max relative error of the log-softmax top-20.
usage: compare_logits.py A.bin B.bin [--vocab 200058]"""
import array, math, sys
V = int(sys.argv[sys.argv.index("--vocab") + 1]) if "--vocab" in sys.argv else 200058
a = array.array("f"); a.frombytes(open(sys.argv[1], "rb").read()); b = array.array("f"); b.frombytes(open(sys.argv[2], "rb").read())
n = min(len(a), len(b)) // V; same = 0; worst = 0.0
for p in range(n):
    x, y = a[p * V:(p + 1) * V], b[p * V:(p + 1) * V]
    ia = max(range(V), key=lambda i: x[i]); ib = max(range(V), key=lambda i: y[i]); same += ia == ib
    mx = max(x); my = max(y); zx = math.log(sum(math.exp(v - mx) for v in x)); zy = math.log(sum(math.exp(v - my) for v in y))
    top = sorted(range(V), key=lambda i: -x[i])[:20]
    worst = max(worst, max(abs((x[i] - mx - zx) - (y[i] - my - zy)) for i in top))
print(f"positions={n} top1 agree={same}/{n} worst |dlogprob| over top-20 = {worst:.4f}")
