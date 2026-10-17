"""MICCAI draft main figure: (a) clean vs attacked across datasets, (b) evidence coverage on TCGA-RCC, (c) RCC accuracy by heterogeneity.
usage: python fig_miccai.py  (reads results/, writes figs/fig_main.pdf)"""
import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); N = f'{B}/results'; os.makedirs(f'{B}/figs', exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
COL = {'full': '#555555', 'admil': '#4C72B0', 'ours': '#C44E52'}
def ms(v): return np.mean(v), np.std(v)
c16 = [json.load(open(f))['slide'] for f in sorted(glob.glob(f'{N}/c16/distill2_b0_s*.json'))]
c17 = [json.load(open(f))['res'] for f in sorted(glob.glob(f'{N}/c17/c17_b0_k4_f*_s*.json'))]
rcc = [json.load(open(f))['res'] for f in sorted(glob.glob(f'{N}/tcga_rcc/*.json'))]
D = [('CAMELYON16\n(BA)', {'full': [r['full_uni']['balacc'] for r in c16], 'admil': [r['admil_K16']['balacc'] for r in c16], 'ours': [r['distill2uni_K16']['balacc'] for r in c16],
                         'admil_att': [r['admil_wb_K16']['balacc'] for r in c16], 'ours_att': [r['distill2uni_adapt_K16']['balacc'] for r in c16]}),
     ('CAMELYON17\n(quadratic κ)', {'full': [r['full_conch']['kappa'] for r in c17], 'admil': [r['admil']['kappa'] for r in c17], 'ours': [r['teach']['kappa'] for r in c17],
                                   'admil_att': [r['admil_wb_worst']['kappa'] for r in c17], 'ours_att': [r['teach_wb_worst']['kappa'] for r in c17]}),
     ('TCGA-RCC\n(BA)', {'full': [r['full_conch']['balacc'] for r in rcc], 'admil': [r['admil']['balacc'] for r in rcc], 'ours': [r['teach']['balacc'] for r in rcc],
                         'admil_att': [r['admil_wb_worst']['balacc'] for r in rcc], 'ours_att': [r['teach_wb_worst']['balacc'] for r in rcc]})]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={'width_ratios': [2.2, 1, 1.2]})
a = ax[0]; w = .15
for j, (name, d) in enumerate(D):
    for k, (key, lab, col, hatch) in enumerate([('full', 'full slide', COL['full'], ''), ('admil', 'single selector', COL['admil'], ''), ('ours', 'ours', COL['ours'], ''),
                                                 ('admil_att', 'single selector, attacked', COL['admil'], '////'), ('ours_att', 'ours, attacked', COL['ours'], '////')]):
        m, s = ms(d[key]); a.bar(j + (k - 2) * w, m, w, yerr=s, color=col if not hatch else 'white', edgecolor=col, hatch=hatch, lw=.6, error_kw={'lw': .5, 'capsize': 1},
                                  label=lab if j == 0 else None)
a.axhline(0, color='k', lw=.4); a.set_xticks(range(3)); a.set_xticklabels([n for n, _ in D]); a.set_ylim(-.45, 1.05); a.set_ylabel('score')
a.legend(fontsize=5.5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.22)); a.set_title('(a) clean vs. compromised selector', fontsize=7)
Z = [np.load(f) for f in sorted(glob.glob(f'{N}/tcga_rcc/*.npz'))]
cat = lambda k: np.concatenate([z[k] for z in Z])
a = ax[1]
for k, (lab, key) in enumerate([('single selector', 'admil'), ('ours', 'ours')]):
    a.bar(k, cat(f'runnerup_{key}').mean(), .6, color=COL[key])
    a.text(k, cat(f'runnerup_{key}').mean() + .02, f"{cat(f'runnerup_{key}').mean():.2f}\n({cat(f'cov_{key}').mean():.1f} subtypes)", ha='center', fontsize=6)
a.set_xticks([0, 1]); a.set_xticklabels(['single\nselector', 'ours']); a.set_ylim(0, 1.15); a.set_ylabel('slides with runner-up evidence')
a.set_title('(b) RCC: evidence coverage', fontsize=7)
a = ax[2]; w = .26  # per-subtype recall, pooled over folds x seeds
for k, (key, lab, col) in enumerate([('full_conch', 'full slide', COL['full']), ('admil', 'single selector', COL['admil']), ('teach', 'ours', COL['ours'])]):
    r = np.array([[(z[key].argmax(1)[z['y'] == c] == c).mean() for c in range(3)] for z in Z])
    a.bar(np.arange(3) + (k - 1) * w, r.mean(0), w, yerr=r.std(0), color=col, label=lab, error_kw={'lw': .5, 'capsize': 1})
a.set_xticks(range(3)); a.set_xticklabels(['ccRCC\n(n=492)', 'pRCC\n(n=283)', 'chRCC\n(n=107)']); a.set_ylim(.75, 1.0); a.set_ylabel('recall')
a.set_title('(c) RCC: recall per subtype', fontsize=7)
plt.tight_layout(); plt.savefig(f'{B}/figs/fig_main.pdf', bbox_inches='tight'); plt.savefig(f'{B}/figs/fig_main.png', dpi=200, bbox_inches='tight')
print('saved')
