"""MICCAI results figure: clean vs worst case under one compromised slot, per presenter and cohort (mean ± sd over 3 seeds).
Reads results/ext/miccai_table.json (written by miccai_table.py). usage: python analysis/fig_miccai.py -> paper/miccai/fig_results.pdf"""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); T = json.load(open(f'{B}/results/ext/miccai_table.json'))
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
P = [('single selector', 'single\nsel.', '#888888'), ('advocates', 'adv.', '#d08c60'), ('advocates + selector slot ($K$)', 'adv.\n+sel.', '#c9b458'),
     ('redundancy, isolated', 'red.\niso.', '#6a9fd0'), ('\\textbf{redundancy, joint}', 'red.\njoint', '#2f5f98')]
DS = [('c17', 'CAMELYON17 ($\\kappa$)'), ('tcga_rcc', 'TCGA-RCC (bal. acc.)'), ('tcga_nsclc', 'TCGA-NSCLC (bal. acc.)')]
FULL = {'c17': .890, 'tcga_rcc': .933, 'tcga_nsclc': .923}
fig, ax = plt.subplots(1, 3, figsize=(7, 1.75), sharey=True); x = np.arange(len(P)); w = .38
for a, (ds, title) in zip(ax, DS):
    for i, (k, lab, col) in enumerate(P):
        (cm, cs), (wm, ws) = T[ds][k]
        a.bar(i - w / 2, cm, w, yerr=cs, color=col, alpha=.35, edgecolor=col, lw=.6, error_kw=dict(lw=.5, capsize=1))
        a.bar(i + w / 2, wm, w, yerr=ws, color=col, edgecolor=col, lw=.6, error_kw=dict(lw=.5, capsize=1))
        a.text(i + w / 2, max(wm, 0) + ws + .025, ('%.2f' % wm).replace('0.', '.'), ha='center', fontsize=5.5)
    a.axhline(FULL[ds], ls=':', c='k', lw=.6); a.axhline(0, c='k', lw=.5)
    a.set_title(title, fontsize=7); a.set_xticks(x); a.set_xticklabels([p[1] for p in P], fontsize=5.5); a.set_ylim(-.3, 1.02)
ax[0].set_ylabel('score'); ax[1].bar(0, 0, color='#888888', alpha=.35, label='clean'); ax[1].bar(0, 0, color='#888888', label='worst case')
ax[1].plot([], [], ls=':', c='k', lw=.6, label='full-slide MIL'); ax[1].legend(fontsize=5.5, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.03), ncol=3, columnspacing=1, handlelength=1.5)
plt.tight_layout(pad=.3); plt.savefig(f'{B}/paper/miccai/fig_results.pdf'); plt.savefig(f'{B}/paper/figs/fig_miccai_results.png', dpi=200)
