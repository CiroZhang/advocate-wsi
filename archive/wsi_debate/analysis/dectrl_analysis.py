import json, sys, numpy as np
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]; nat = np.array([r['native']['pred'] == r['y'] for r in rs]); print("##", f.split("/")[-1], len(rs), "native", nat.mean().round(3))
    for s in rs[0]['subst']:
        sp = np.array([r['subst'][s]['pred'] == r['y'] for r in rs])
        print(f"   {s:10s} acc {sp.mean():.3f}  c->w {(~sp[nat]).mean():.3f}  w->c {sp[~nat].mean():.3f}")
