"""MICCAI main table: pathology presenters, clean and worst case under one compromised slot; mean ± sd over 3 seeds (each the mean
of 5 folds). Sources: *_ext_atty.json (advocates, hybrid; attacker may take any slot incl. the true class's), *_ext_red.json
(redundancy), *_budget.json (single selector at 1-3x budget), *_ext.json (weak honest scorer). Worst = adaptive (greedy set) attack,
or min(per-tile, adaptive) aggregate for x_red. usage: python analysis/miccai_table.py -> paper/miccai/tab_main.tex + stdout"""
import os, json, glob, re, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); R = f'{B}/results/ext'
DS = [('c17', 'kappa'), ('tcga_rcc', 'balacc'), ('tcga_nsclc', 'balacc')]
def load(ds, suf):
    out = {}
    for f in glob.glob(f'{R}/{ds}_b0_k4_f*_s*_{suf}.json'):
        fo, s = map(int, re.search(r'_f(\d)_s(\d)_', f).groups())
        for k, v in json.load(open(f))['res'].items():
            mm = re.match(r'x_hyb(\d+)(_\w+)?$', k)
            if mm: k = ('hybKc' if mm.group(1) == '4' else 'hybK') + (mm.group(2) or '')
            mm = re.match(r'admil_b(\d+)(_wb_worst)?$', k)
            if mm: k = 'bud%d' % int(mm.group(1)) + (mm.group(2) or '')
            out.setdefault(k, {}).setdefault(s, {})[fo] = v
    return out
def stat(D, k, met):
    if k not in D: return None
    per = [np.mean([v[met] for v in D[k][s].values()]) for s in sorted(D[k])]; return np.mean(per), np.std(per)
ROWS = [('single selector', 'admil', 'admil_wb_worst'), ('\\quad budget $K{+}K_c$', 'bud_mid', 'bud_mid_wb_worst'), ('\\quad budget $2K$', 'bud_hi', 'bud_hi_wb_worst'),
        ('\\quad weak honest scorer (10\\% data)', 'admil_weak', None), ('advocates', 'x_base', 'x_base_ad'), ('\\quad independent scorers', 'x_indep', 'x_indep_wb'),
        ('redundancy, joint', 'x_red', 'x_red_min'), ('redundancy, isolated', 'x_rediso', 'x_rediso_ad'), ('advocates + neutral, isolated', 'x_hybiso', 'x_hybiso_ad'),
        ('advocates + neutral ($K_c$)', 'hybKc', 'hybKc_ad'), ('advocates + neutral ($K$)', 'hybK', 'hybK_ad')]
T = {}
for ds, met in DS:
    D = {}
    for suf in ('ext', 'ext_atty', 'ext_red', 'budget'):
        for k, v in load(ds, suf).items():
            if suf == 'ext' and k not in ('admil_weak',): continue
            if suf == 'budget' and not k.startswith('bud'): continue
            D[k] = v
    bs = sorted({int(re.match(r'bud(\d+)$', k).group(1)) for k in D if re.match(r'bud\d+$', k)})
    for tag, b in (('bud_mid', bs[1]), ('bud_hi', bs[2])): D[tag], D[tag + '_wb_worst'] = D[f'bud{b}'], D[f'bud{b}_wb_worst']
    D['x_red_min'] = {s: {f: {met: min(D['x_red_wb'][s][f][met], D['x_red_ad'][s][f][met])} for f in D['x_red_ad'][s]} for s in D['x_red_ad']}
    T[ds] = {r[0]: (stat(D, r[1], met), stat(D, r[2], met) if r[2] else None) for r in ROWS}
f = lambda x: '--' if x is None else ('%.3f' % x[0]).replace('0.', '.').replace('-.', '$-$.') + '\\pmsd{%s}' % ('%.3f' % x[1]).replace('0.', '.')
L = [' & '.join([name] + [f(T[ds][name][j]) for ds, _ in DS for j in (0, 1)]) + '\\\\' for name, *_ in ROWS]
open(f'{B}/paper/miccai/tab_main.tex', 'w').write('\n'.join(L) + '\n\\bottomrule\n')
for name, *_ in ROWS: print(f'{name:45s}', '  '.join('%.3f/%s' % (T[ds][name][0][0], '%.3f' % T[ds][name][1][0] if T[ds][name][1] else ' -- ') for ds, _ in DS))
