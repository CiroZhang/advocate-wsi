"""Isolate-then-aggregate redundancy baselines on natural images, computed exactly from cached per-tile scores S (overview + one tile).
Each slot's tile is judged separately (S[t]); slots are aggregated by majority vote (ties -> summed log-probs) or by mean log-prob.
Attacker: replaces ONE slot by ANY tile of the image (exhaustive, label-aware). Reported: clean acc, worst-case acc.
Slot sets: 'redund' = C+1 most confident tiles (plain redundancy, no labels); 'hybrid' = per-option best tile + most confident tile.
usage: python analysis/redundancy_offline.py"""
import json, numpy as np
def agg(P, mode):  # P: (m, C) log-probs per slot
    if mode == 'mean': return int(P.mean(0).argmax())
    v = np.bincount(P.argmax(1), minlength=P.shape[1]).astype(float); v += 1e-3 * P.sum(0) / (abs(P.sum(0)).max() + 1e-9); return int(v.argmax())
def run(tag):
    R = [json.loads(l) for l in open(f'results/vlm/{tag}_partial.jsonl')]; out = {}
    for name in ['redund', 'hybrid']:
        for mode in ['vote', 'mean']:
            cl, wc = [], []
            for r in R:
                S = np.array(r['S_glob']); y = r['y']; n = r['nopt']; S = S[:, :n] if S.shape[1] > n else S
                conf = np.exp(S).max(1)
                sl = [int(t) for t in np.argsort(-conf)[:n + 1]] if name == 'redund' else [int(S[:, k].argmax()) for k in range(n)] + [int(conf.argmax())]
                P = S[sl]; ok = agg(P, mode) == y; cl.append(ok); bad = False
                for j in range(len(sl)):
                    for t in range(len(S)):
                        Q = P.copy(); Q[j] = S[t]
                        if agg(Q, mode) != y: bad = True; break
                    if bad: break
                wc.append(ok and not bad)
            out[f'{name}_{mode}'] = (round(np.mean(cl), 3), round(np.mean(wc), 3))
    return out
for t in ['vstar', 'vstar_q3', 'hr4k_q25']: print(t, run(t))
