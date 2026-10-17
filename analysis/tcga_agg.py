"""Aggregate TCGA subtyping runs (tcga_{CO}_{CH}_k{Kc}_f*_s*.json/.npz): per-method mean over folds/seeds, accuracy by
heterogeneity tercile, evidence coverage, and paired per-run deltas for the key contrasts.
usage: python tcga_agg.py DIR rcc [b0] [4]"""
import sys, glob, json, numpy as np
D, CO = sys.argv[1], sys.argv[2]; CH = sys.argv[3] if len(sys.argv) > 3 else 'b0'; KC = sys.argv[4] if len(sys.argv) > 4 else '4'
fs = sorted(glob.glob(f'{D}/tcga_{CO}_{CH}_k{KC}_f*_s*.json')); R = [json.load(open(f)) for f in fs]
print(len(fs), 'runs', CO, CH, 'Kc', KC, '| mean tiles/slide', np.mean([r['info']['mean_tiles_test'] for r in R]).round(0))
KS = ['full_conch', 'teach', 'noflag', 'admil', 'percls_full', 'teach_wb_worst', 'noflag_wb_worst', 'admil_wb_worst']
for m in ['acc', 'balacc', 'auc']:
    print(f'== {m}')
    for k in KS:
        v = [r['res'][k][m] for r in R if k in r['res']]
        if v: print(f'  {k:16s} {np.mean(v):.3f} ± {np.std(v):.3f}')
print('== acc by heterogeneity tercile (low / mid / high)')
for k in ['full_conch', 'teach', 'noflag', 'admil', 'percls_full']:
    v = np.array([r['res'][k]['acc_by_het'] for r in R if 'acc_by_het' in r['res'].get(k, {})])
    if len(v): print(f'  {k:16s}', ' '.join(f'{x:.3f}' for x in np.nanmean(v, 0)))
def paired(a, b, m='acc'):
    d = np.array([r['res'][a][m] - r['res'][b][m] for r in R])
    print(f'  {a} - {b} ({m}): {d.mean():+.3f}  wins {int((d > 0).sum())}/{len(d)}  ties {int((d == 0).sum())}')
print('== paired contrasts')
for a, b in [('percls_full', 'admil'), ('teach', 'admil'), ('teach', 'full_conch'), ('teach', 'noflag'), ('teach_wb_worst', 'noflag_wb_worst'), ('teach_wb_worst', 'admil_wb_worst')]:
    for m in ['acc', 'balacc']: paired(a, b, m)
Z = [np.load(f.replace('.json', '.npz')) for f in fs]
if 'cov_ours' in Z[0]:
    print('== evidence coverage (per slide, pooled)')
    for k in ['cov_admil', 'cov_ours', 'runnerup_admil', 'runnerup_ours', 'het']: print(f'  {k:16s} {np.mean(np.concatenate([z[k] for z in Z])):.3f}')
