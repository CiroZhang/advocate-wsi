"""Why does isolation work? Exact one-slot worst case (any grid tile) for isolate-then-average presenters, from cached scores.
  glob_m : m most confident tiles, each judged WITH the overview (S_glob), mean log-prob   [the defence]
  alone_m: same tiles judged WITHOUT the overview (S_alone), mean log-prob
  alone_m+ov: S_alone tiles plus the overview-only prediction as one extra (honest) voter
usage: python analysis/isolation_ablation.py"""
import json, numpy as np
def worst(P, S, y, honest=None):  # P: slot log-probs (m, C); S: all tiles' log-probs; honest: extra fixed slot
    f = lambda Q: (np.vstack([Q, honest]) if honest is not None else Q).mean(0).argmax() == y
    if not f(P): return 0, 0
    for j in range(len(P)):
        for t in range(len(S)):
            Q = P.copy(); Q[j] = S[t]
            if not f(Q): return 1, 0
    return 1, 1
for tag in ['vstar', 'vstar_q3', 'hr4k_q25']:
    R = [json.loads(l) for l in open(f'results/vlm/{tag}_partial.jsonl')]; out = {}
    for m in [1, 2, 3]:
        for nm in ['glob', 'alone', 'alone+ov']:
            res = []
            for r in R:
                Sg = np.array(r['S_glob']); Sa = np.array(r['S_alone']); y = r['y']; G = np.array(r['base']['global_only'])
                Sx = Sg if nm == 'glob' else Sa; conf = np.exp(Sx).max(1); sl = list(np.argsort(-conf)[:m])
                res.append(worst(Sx[sl], Sx, y, G if nm == 'alone+ov' else None))
            a = np.array(res).mean(0); out[f'{nm}{m}'] = f'{a[0]:.3f}/{a[1]:.3f}'
    print(tag, out)
