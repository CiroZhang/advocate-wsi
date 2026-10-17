"""Accuracy vs. per-slide compute for every method and cancer type (mean over runs).
Compute model (A100, seconds/slide, measured during feature extraction / io_bench; tiling = tissue mask + grid):
  full slide        tiling + full-res read(all) + FM(all)
  cheap full slide  tiling + low-mag read(all) + B0(all)
  random K          tiling + K * per-tile(full-res read + FM)
  ADMIL / ours      tiling + low-mag read(all) + B0(all) + K * per-tile(full-res read + FM)
  + escalate f      above + f * (full-res read(all) + FM(all))     [lowest-confidence fraction f -> full-slide model]
usage: python compute_table.py  (reads ../results, writes ../results/compute_table.{md,json})"""
import os, json, glob, numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
TIM = {  # tiling, full-res read, low-mag read, FM, B0, tiles  (C16: tiles pre-extracted, tiling not measured)
    'C16': dict(tile=0.0, l0=22.49, l2=3.93, fm=10.80, b0=1.50, n=12182, K=16, fm_name='UNI'),
    'C17': dict(tile=2.67, l0=19.36, l2=2.57, fm=2.41, b0=1.33, n=2090, K=16, fm_name='CONCH'),
    'RCC': dict(tile=1.62, l0=25.97, l2=4.72, fm=4.77, b0=1.48, n=4453, K=12, fm_name='CONCH'),
    'NSCLC': dict(tile=1.37, l0=20.73, l2=4.05, fm=4.22, b0=1.16, n=3959, K=8, fm_name='CONCH'),
}
TYPE = {'C16': 'breast LN metastasis (2-class)', 'C17': 'breast LN stage (4-class, ordinal)', 'RCC': 'kidney RCC subtype (3-class)', 'NSCLC': 'lung NSCLC subtype (2-class)'}
MET = {'C16': 'BA', 'C17': 'kappa', 'RCC': 'BA', 'NSCLC': 'BA'}
def cost(t, kind, K=None, f=0.0):
    per = (t['l0'] + t['fm']) / t['n']
    if kind == 'full': return t['tile'] + t['l0'] + t['fm']
    if kind == 'cheap': return t['tile'] + t['l2'] + t['b0']
    if kind == 'rand': return t['tile'] + K * per
    return t['tile'] + t['l2'] + t['b0'] + K * per + f * (t['l0'] + t['fm'])
def score(ds, y, lg):
    p = (lg > 0).astype(int) if lg.ndim == 1 else lg.argmax(1)
    return cohen_kappa_score(y, p, weights='quadratic') if ds == 'C17' else balanced_accuracy_score(y, p)
def conf(lg):
    if lg.ndim == 1: return np.abs(lg)
    e = np.exp(lg - lg.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True); s = np.sort(p, 1); return s[:, -1] - s[:, -2]
def escalate(y, lg, full, f):
    k = int(round(f * len(y))); idx = np.argsort(conf(lg))[:k]; out = lg.copy(); out[idx] = full[idx]; return out
def runs(ds):  # list of dicts key -> logits, with 'y'
    if ds == 'C16':
        return [{'y': d['y'], 'full': d['full_uni'], 'admil': d['admil_K16'], 'ours': d['distill2uni_K16'], 'admil_att': d['admil_wb_K16'], 'ours_att': d['distill2uni_adapt_K16']}
                for d in map(np.load, sorted(glob.glob(f'{R}/c16/distill2_b0_logits_s*.npz')))]
    pat = {'C17': f'{R}/c17/c17_b0_k4_f*_s*.npz', 'RCC': f'{R}/tcga_rcc/tcga_rcc_b0_k4_f*_s*.npz', 'NSCLC': f'{R}/tcga_nsclc/tcga_nsclc_b0_k4_f*_s*.npz'}[ds]
    return [{'y': d['y'], 'full': d['full_conch'], 'admil': d['admil'], 'ours': d['teach'], 'admil_att': d['admil_wb_worst'], 'ours_att': d['teach_wb_worst']}
            for d in map(np.load, sorted(glob.glob(pat)))]
rows, out = [], {}
for ds, t in TIM.items():
    Rn = runs(ds)
    if not Rn: continue
    m = lambda key: np.mean([score(ds, r['y'], r[key]) for r in Rn])
    tab = [('full slide (' + t['fm_name'] + ', all tiles)', m('full'), None, cost(t, 'full'))]
    fr = [json.load(open(f))['res'] for f in glob.glob(f'{R}/frontier/frontier_{ds.lower()}_f*_s*.json')]
    mk = 'kappa' if ds == 'C17' else 'balacc'
    if fr:
        tab.append(('cheap full slide (B0, all tiles)', np.mean([r['cheap_full'][mk] for r in fr]), None, cost(t, 'cheap')))
        for K in [t['K'], 4 * t['K'], 16 * t['K']]:
            tab.append((f'{t["fm_name"]} on {K} random tiles (trained on random)', np.mean([r[f'rand_{K}_tr'][mk] for r in fr]), None, cost(t, 'rand', K)))
    tab.append((f'ADMIL-style ({t["K"]} tiles)', m('admil'), m('admil_att'), cost(t, 'sel', t['K'])))
    tab.append((f'ours ({t["K"]} tiles)', m('ours'), m('ours_att'), cost(t, 'sel', t['K'])))
    for f in [0.1, 0.2]:
        for key, lab in [('admil', 'ADMIL-style'), ('ours', 'ours')]:
            tab.append((f'{lab} + escalate {int(f * 100)}%', np.mean([score(ds, r['y'], escalate(r['y'], r[key], r['full'], f)) for r in Rn]), None, cost(t, 'sel', t['K'], f)))
    out[ds] = [dict(method=a, score=b, attacked=c, sec=d) for a, b, c, d in tab]
    full_s = cost(t, 'full')
    rows.append(f'\n### {ds}: {TYPE[ds]} ({MET[ds]}, {len(Rn)} runs)\n\n| method | {MET[ds]} | attacked | s/slide | speed-up |\n|---|---|---|---|---|')
    rows += [f'| {a} | {b:.3f} | {"" if c is None else f"{c:.3f}"} | {d:.1f} | {full_s / d:.1f}x |' for a, b, c, d in tab]
txt = '\n'.join(rows); print(txt)
open(f'{R}/compute_table.md', 'w').write('# Accuracy vs compute per cancer type\n' + (__doc__ or '') + txt + '\n'); json.dump(out, open(f'{R}/compute_table.json', 'w'), indent=1)
