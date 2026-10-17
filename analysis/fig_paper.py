"""Paper figures (two-column preprint).
  fig_robust.pdf   (column width): clean vs compromised selector on 4 datasets
  fig_frontier.pdf (full width):   score vs per-slide compute per dataset (focal: C16, C17; diffuse: RCC, NSCLC)
usage: python compute_table.py && python fig_paper.py   (reads ../results, writes ../paper/figs)"""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); O = f'{B}/paper/figs'; os.makedirs(O, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
COL = {'full': '#555555', 'admil': '#4C72B0', 'ours': '#C44E52', 'rand': '#55A868', 'cheap': '#CCB974'}
T = json.load(open(f'{B}/results/compute_table.json'))
NAME = {'C16': 'CAMELYON16', 'C17': 'CAMELYON17', 'RCC': 'TCGA-RCC', 'NSCLC': 'TCGA-NSCLC'}
MET = {'C16': 'BA', 'C17': 'quadratic κ', 'RCC': 'BA', 'NSCLC': 'BA'}
DS = ['C16', 'C17', 'RCC', 'NSCLC']
def row(ds, pre): return next(r for r in T[ds] if r['method'].startswith(pre))
# ---- robustness
fig, a = plt.subplots(figsize=(3.4, 1.9)); w = .16
for j, ds in enumerate(DS):
    vals = [(row(ds, 'full slide')['score'], COL['full'], '', 'full slide'), (row(ds, 'ADMIL-style (')['score'], COL['admil'], '', 'single selector'),
            (row(ds, 'ours (')['score'], COL['ours'], '', 'ours'), (row(ds, 'ADMIL-style (')['attacked'], COL['admil'], '////', 'single selector, attacked'),
            (row(ds, 'ours (')['attacked'], COL['ours'], '////', 'ours, attacked')]
    for k, (v, c, h, lab) in enumerate(vals):
        a.bar(j + (k - 2) * w, v, w, color=c if not h else 'white', edgecolor=c, hatch=h, lw=.6, label=lab if j == 0 else None)
a.axhline(0, color='k', lw=.4); a.set_xticks(range(4)); a.set_xticklabels([f'{NAME[d]}\n({MET[d]})' for d in DS], fontsize=6); a.set_ylim(-.35, 1.05); a.set_ylabel('score')
a.legend(fontsize=5.5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.25))
plt.tight_layout(); plt.savefig(f'{O}/fig_robust.pdf', bbox_inches='tight'); plt.savefig(f'{O}/fig_robust.png', dpi=200, bbox_inches='tight'); plt.close()
# ---- frontier
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9)); fig.subplots_adjust(wspace=.45)
for a, ds in zip(ax, DS):
    R = T[ds]; full = row(ds, 'full slide'); ch = row(ds, 'cheap full'); rnd = [r for r in R if 'random tiles' in r['method']]
    a.plot([r['sec'] for r in rnd], [r['score'] for r in rnd], '-o', ms=2.5, color=COL['rand'], lw=.8, label='FM on random tiles (K, 4K, 16K)')
    a.scatter(ch['sec'], ch['score'], marker='s', s=12, color=COL['cheap'], label='cheap MIL, all tiles', zorder=3)
    a.scatter(full['sec'], full['score'], marker='*', s=30, color=COL['full'], label='full slide', zorder=3)
    for key, pre in [('admil', 'ADMIL-style'), ('ours', 'ours')]:
        pts = [row(ds, f'{pre} ('), row(ds, f'{pre} + escalate 10%'), row(ds, f'{pre} + escalate 20%')]
        a.plot([p['sec'] for p in pts], [p['score'] for p in pts], '-^' if key == 'admil' else '-D', ms=2.5, lw=.8, color=COL[key],
               label=('single selector' if key == 'admil' else 'ours') + ' (+10/20% escalated)')
    a.set_xscale('log'); a.set_xlim(min(r['sec'] for r in R if r['sec'] > 0) / 1.6, full['sec'] * 1.8); a.set_xlabel('s / slide'); a.set_title(f"{NAME[ds]} ({'focal' if ds in ('C16', 'C17') else 'diffuse'})", fontsize=7)
    lo = min(r['score'] for r in R if r['score'] is not None); a.set_ylim(max(lo - .03, .5), 1.0); a.set_ylabel(MET[ds], labelpad=1)
ax[0].legend(fontsize=5.5, ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0, -.3))
plt.savefig(f'{O}/fig_frontier.pdf', bbox_inches='tight'); plt.savefig(f'{O}/fig_frontier.png', dpi=200, bbox_inches='tight')
print('saved')
