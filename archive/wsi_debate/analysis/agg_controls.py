import json, glob, numpy as np, sys
fs = sorted(glob.glob(sys.argv[1] + "/controls_s*.json")); R = [json.load(open(f)) for f in fs]; print(len(R), "seeds")
def agg(sec):
    keys = sorted(set().union(*[r[sec].keys() for r in R]), key=lambda s: (len(s), s))
    for k in keys:
        rows = [r[sec][k] for r in R if k in r[sec]]; mets = rows[0].keys()
        print(f"[{sec}] {k} (n={len(rows)})"); [print(f"   {m:42s} {np.mean([x[m] for x in rows]):.3f} ± {np.std([x[m] for x in rows]):.3f}") for m in mets]
for s in ("mech", "taxonomy", "sparsity"): agg(s)
