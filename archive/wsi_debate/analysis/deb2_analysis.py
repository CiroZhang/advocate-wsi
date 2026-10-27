import json, numpy as np, sys, glob
R0 = np.random.default_rng(0); am = lambda v: int(np.argmax(v))
def ci(x):
    x = np.asarray(x, float); bs = [x[R0.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():.3f} [{np.quantile(bs,.025):.3f},{np.quantile(bs,.975):.3f}]"
for f in sorted(glob.glob(sys.argv[1] + "/*_deb2.jsonl")):
    rs = [json.loads(l) for l in open(f)]; print("##", f.split("/")[-1], len(rs))
    for k in ("nway", "top2", "coop2", "coopn", "att2", "attn"): print(f"   {k:6s} {ci([am(r[k]) == r['y'] for r in rs])}")
    print("   top2 contains truth", ci([r['top2_has_y'] for r in rs]))
