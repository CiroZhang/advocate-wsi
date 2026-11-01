"""MICCAI Fig. 3: (a-c) CAMELYON17 pN-stage confusion (row-normalised, all 5 folds x 3 seeds) for the single selector clean and under
attack and for advocates + selector slot (K) under attack; (d) TCGA-RCC: % of slides of each subtype containing >= K tiles that the
full-slide model reads as another subtype (results/ext/mimic_stats.json). usage: python analysis/fig_clinical.py -> paper/miccai/fig_clinical.pdf"""
import os, glob, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
plt.rcParams.update({'font.size': 6.5, 'pdf.fonttype': 42})
def conf(k):
    M = np.zeros((4, 4))
    for f in glob.glob(f'{B}/results/ext/c17_b0_k4_f*_s*_ext_atty.npz'):
        z = np.load(f); np.add.at(M, (z['y'], z[k].argmax(1)), 1)
    return M / M.sum(1, keepdims=True)
ST = ['neg.', 'ITC', 'micro', 'macro']
P = [('admil', '(a) single selector, clean'), ('admil_wb_worst', '(b) single selector, attacked'), ('x_hyb16_ad', '(c) adv. + selector slot, attacked')]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
def heat(a, M, xl, yl, title, xlab, ylab):
    a.imshow(M, cmap='Blues', vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]): a.text(j, i, '%d' % round(100 * M[i, j]), ha='center', va='center', fontsize=6, color='w' if M[i, j] > .6 else 'k')
    a.set_xticks(range(len(xl))); a.set_xticklabels(xl, fontsize=5.5); a.set_yticks(range(len(yl))); a.set_yticklabels(yl, fontsize=5.5)
    a.set_title(title, fontsize=6.5); a.set_xlabel(xlab, fontsize=6); a.set_ylabel(ylab, fontsize=6)
for i, (a, (k, t)) in enumerate(zip(ax, P)): heat(a, conf(k), ST, ST, t, 'predicted', 'true' if i == 0 else '')
d = json.load(open(f'{B}/results/ext/mimic_stats.json'))['rcc']['per_class_ge_K']; R = ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH']
M = np.array([[np.nan if r == c else d[r][c] for c in R] for r in R]); L = ['clear', 'papill.', 'chromo.']
heat(ax[3], M, L, L, '(d) RCC: $\\geq$K mimic tiles', 'tiles read as', 'true')
plt.tight_layout(pad=.3, w_pad=.8); plt.savefig(f'{B}/paper/miccai/fig_clinical.pdf'); plt.savefig(f'{B}/paper/figs/fig_miccai_clinical.png', dpi=200)
