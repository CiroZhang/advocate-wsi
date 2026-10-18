"""Paired slide-level bootstrap for the main comparisons (95% CI of the score difference, one-sided p).
Test slides are resampled within each fold (same draw for every seed of that fold); the statistic is the mean over runs,
matching Table 1. CAMELYON16 has one test set (3 seeds).
usage: python stats.py   (reads ../results, writes ../results/stats.{md,json})"""
import os, json, glob, re, numpy as np
from compute_table import score, R
B = 2000; rng = np.random.default_rng(0)
PAIRS = [('ours_att', 'admil_att', 'attacked: ours - single selector'),
         ('ours', 'admil', 'clean: ours - single selector'),
         ('ours', 'full', 'clean: ours - full slide'),
         ('ours_att', 'noflag_att', 'attacked: ours - no presenter flags')]
def load(ds):  # fold -> list of per-seed dicts
    if ds == 'C16':
        K = dict(full='full_uni', admil='admil_K16', ours='distill2uni_K16', admil_att='admil_wb_K16', ours_att='distill2uni_adapt_K16', noflag_att='noflag_adapt_K16')
        fs = {0: sorted(glob.glob(f'{R}/c16/distill2_b0_logits_s*.npz'))}
    else:
        K = dict(full='full_conch', admil='admil', ours='teach', admil_att='admil_wb_worst', ours_att='teach_wb_worst', noflag_att='noflag_wb_worst')
        pat = {'C17': 'c17/c17_b0_k4', 'RCC': 'tcga_rcc/tcga_rcc_b0_k4', 'NSCLC': 'tcga_nsclc/tcga_nsclc_b0_k4'}[ds]
        fs = {}
        for f in sorted(glob.glob(f'{R}/{pat}_f*_s*.npz')): fs.setdefault(int(re.search(r'_f(\d+)_', f)[1]), []).append(f)
    out = {}
    for fo, L in fs.items():
        runs = []
        for f in L:
            d = np.load(f); runs.append({'y': d['y'], **{k: d[v] for k, v in K.items() if v in d}})
        out[fo] = runs
    return out
def stat(ds, F, a, b, idx=None):
    v = []
    for fo, runs in F.items():
        i = slice(None) if idx is None else idx[fo]
        v += [score(ds, r['y'][i], r[a][i]) - score(ds, r['y'][i], r[b][i]) for r in runs]
    return np.mean(v)
def main():
    res, md = {}, ['| dataset | comparison | diff | 95% CI | p (one-sided) |', '|---|---|---|---|---|']
    for ds in ['C16', 'C17', 'RCC', 'NSCLC']:
        F = load(ds); n = {fo: len(r[0]['y']) for fo, r in F.items()}
        draws = [{fo: rng.integers(0, n[fo], n[fo]) for fo in F} for _ in range(B)]
        for a, b, lab in PAIRS:
            if not all(a in r and b in r for runs in F.values() for r in runs): continue
            d0 = stat(ds, F, a, b); bs = np.array([stat(ds, F, a, b, ix) for ix in draws])
            lo, hi = np.percentile(bs, [2.5, 97.5]); p = (np.sum(bs <= 0) + 1) / (B + 1) if d0 > 0 else (np.sum(bs >= 0) + 1) / (B + 1)
            res.setdefault(ds, []).append(dict(comparison=lab, diff=d0, lo=lo, hi=hi, p=p))
            md.append(f'| {ds} | {lab} | {d0:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} |')
            print(md[-1], flush=True)
    open(f'{R}/stats.md', 'w').write('# Paired bootstrap\n' + __doc__ + '\n' + '\n'.join(md) + '\n'); json.dump(res, open(f'{R}/stats.json', 'w'), indent=1)
if __name__ == '__main__': main()
