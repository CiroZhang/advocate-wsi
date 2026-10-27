"""Pathology redundancy baselines (RED=1 runs) vs hybrid, plus single selector at matched budget (BONLY runs).
Per dataset: mean over 3 seeds x 5 folds; paired difference vs hybrid-K adaptive with a 95% bootstrap CI over the 15 fold-seed runs.
x_red = C+1 unlabelled joint slots (attacker may take any slot); x_rediso / x_hybiso = each slot judged alone, mean log-prob.
usage: python analysis/red_summary.py -> results/ext/SUMMARY_red.md"""
import os, json, glob, re, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); R = f'{B}/results/ext'
def load(ds, suf):
    out = {}
    for f in glob.glob(f'{R}/{ds}_b0_k4_f*_s*_{suf}.json'):
        fs = re.search(r'_f(\d)_s(\d)_', f).groups(); res = json.load(open(f))['res']
        for k, v in res.items():
            mm = re.match(r'x_hyb(\d+)(_\w+)?$', k)
            if mm and mm.group(1) != '4': k = 'hybK' + (mm.group(2) or '')
            out.setdefault(k, {})[fs] = v
    return out
def ci(d, rng=np.random.default_rng(0)):
    b = [rng.choice(d, len(d)).mean() for _ in range(5000)]; return d.mean(), *np.percentile(b, [2.5, 97.5])
L = ['# Pathology: redundancy baselines vs hybrid (3 seeds x 5 folds)', '',
     'worst = min(per-tile wb, adaptive) of the aggregate metric where both exist. Δ = row − hybrid-K (paired over 15 runs, 95% bootstrap CI).', '']
for ds, met in [('c17', 'kappa'), ('tcga_rcc', 'balacc'), ('tcga_nsclc', 'balacc')]:
    A = load(ds, 'ext_atty'); Rd = load(ds, 'ext_red'); Bd = load(ds, 'budget'); D = {**Rd, **Bd, **{k: v for k, v in A.items() if k.startswith('hybK') or k in ('admil', 'admil_wb_worst', 'x_base', 'x_base_ad')}}
    runs = sorted(D['hybK'])
    g = lambda k: np.array([D[k][r][met] for r in runs]) if k in D and all(r in D[k] for r in runs) else None
    def worst(k):
        xs = [g(k + s) for s in ('_wb', '_ad', '_wb_worst') if g(k + s) is not None]; return np.min(xs, 0) if xs else None
    rows = [('single selector K', 'admil', 'admil_wb_worst')] + [(f'single selector, budget {b}', f'admil_b{b}', f'admil_b{b}_wb_worst') for b in sorted({int(m) for k in D for m in re.findall(r'admil_b(\d+)$', k)})] + [
        ('advocates', 'x_base', 'x_base_ad'), ('redundancy, joint any-slot', 'x_red', None), ('redundancy, isolated', 'x_rediso', 'x_rediso_ad'),
        ('hybrid, isolated', 'x_hybiso', 'x_hybiso_ad'), ('**hybrid-K (ours)**', 'hybK', 'hybK_ad')]
    hc, hw = g('hybK'), g('hybK_ad')
    L += [f'## {ds} ({met}, n={len(runs)})', '', '| presenter | clean | worst | Δclean | Δworst |', '|---|---|---|---|---|']
    for name, kc, kw in rows:
        c = g(kc); w = g(kw) if kw else worst(kc)
        if c is None: continue
        fm = lambda x: '%.3f [%.3f, %.3f]' % ci(x)
        L.append(f'| {name} | {c.mean():.3f} | {w.mean() if w is not None else float("nan"):.3f} | {fm(c - hc)} | {fm(w - hw) if w is not None else ""} |')
    L.append('')
open(f'{R}/SUMMARY_red.md', 'w').write('\n'.join(L)); print('\n'.join(L))
