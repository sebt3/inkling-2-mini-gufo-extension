#!/usr/bin/env python3
"""Compare two tensor dump directories (oracle_dump / inkling_ref --dump): token-major f32 files named <name>.bin.
usage: compare_dumps.py DIR_A DIR_B [DIM]   (DIM: row width override; default inferred from the first file)
Prints per tensor the relative L2 error ||a-b|| / ||b|| over all tokens and its worst token."""
import array, math, os, re, sys
def load(p):
    a = array.array("f"); a.frombytes(open(p, "rb").read()); return a
def key(n):
    m = re.match(r"(.*?)(?:-(-?\d+))?\.bin$", n); return (m.group(1), int(m.group(2)) if m.group(2) is not None else -1)
PER_EXPERT = {"ffn_moe_gate_up_scaled": 2304, "ffn_moe_situ": 1152, "ffn_moe_down_scaled": 2304, "ffn_moe_weights_scaled": 1}
def sorted_rows(x, ids, w):
    """reorder the 8 expert rows of one token by expert id"""
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    out = array.array("f")
    for i in order: out.extend(x[i * w:(i + 1) * w])
    return out
def main():
    A, B = sys.argv[1], sys.argv[2]
    names = sorted((n for n in os.listdir(A) if n.endswith(".bin") and os.path.exists(os.path.join(B, n)) and not n.startswith("logits")), key=key)
    for n in names:
        a, b = load(os.path.join(A, n)), load(os.path.join(B, n))
        if len(a) != len(b): print(f"{n:22s} SIZE MISMATCH {len(a)} vs {len(b)}"); continue
        nm, sl = key(n)
        width = {"Qcur": 3072, "Kcur": 1024, "Vcur": 1024, "kqv_out": 3072, "ffn_moe_logits": 128, "ffn_moe_gate_up_scaled": 8 * 2304, "ffn_moe_situ": 8 * 1152, "ffn_moe_down_scaled": 8 * 2304, "ffn_moe_weights_scaled": 8}.get(nm, 2304)
        ntok = len(a) // width; worst = (0.0, 0); num = den = 0.0
        ida = idb = None
        if nm in PER_EXPERT:
            ida, idb = load(os.path.join(A, f"ffn_moe_topk-{sl}.bin")), load(os.path.join(B, f"ffn_moe_topk-{sl}.bin"))
        for t in range(ntok):
            x, y = a[t * width:(t + 1) * width], b[t * width:(t + 1) * width]
            if ida is not None:
                w = PER_EXPERT[nm]; x = sorted_rows(x, ida[t * 8:(t + 1) * 8], w); y = sorted_rows(y, idb[t * 8:(t + 1) * 8], w)
                if sorted(ida[t * 8:(t + 1) * 8]) != sorted(idb[t * 8:(t + 1) * 8]): print(f"   {n} tok{t}: expert sets differ {sorted(ida[t*8:(t+1)*8])} vs {sorted(idb[t*8:(t+1)*8])}")
            d = sum((p - q) ** 2 for p, q in zip(x, y)); e = sum(q * q for q in y)
            num += d; den += e; r = math.sqrt(d / e) if e else 0.0
            if r > worst[0]: worst = (r, t)
        print(f"{n:22s} rel_err={math.sqrt(num / den) if den else 0:.5f}  worst={worst[0]:.5f}@tok{worst[1]}")
main()
