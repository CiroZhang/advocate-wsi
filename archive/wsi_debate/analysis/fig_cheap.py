"""Fig + table: cheap students (ResNet50-BT) present patches, only presented patches are encoded with UNI for the teacher.
BA vs encoder cost per slide (CAMELYON16 test, 3 seeds). Cost = A100 encoder seconds per slide from enc_timing.json
(N patches through the student encoder + B patches through UNI); falls back to GFLOPs if timing is missing."""
import json, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = 'narval_results/cheap'; KS = [2, 4, 8, 16]
J = [json.load(open(f'{D}/cheap_s{s}.json')) for s in range(6)]
N = J[0]['info']['mean_patches_test']
m = lambda k: np.mean([j['slide'][k]['balacc'] for j in J]); sd = lambda k: np.std([j['slide'][k]['balacc'] for j in J])
T = json.load(open(f'{D}/enc_timing.json')) if os.path.exists(f'{D}/enc_timing.json') else None
if T: cu, cr, unit = 1 / T['UNI_ViT-L16@224']['patches_per_s'], 1 / T['ResNet50@224']['patches_per_s'], 'A100 encoder seconds per slide'
else: cu, cr, unit = 61.6e-3, 4.1e-3, 'encoder TFLOPs per slide'
cost = {'full_uni': N * cu, 'full_r50': N * cr, 'teach_uni': N * cu, 'teach_r50': N * cr}
rows = [('Full-slide ABMIL, UNI', 'full_uni', None), ('Full-slide ABMIL, ResNet50', 'full_r50', None)]
print(f'N={N:.0f}  unit={unit}  full UNI {cost["full_uni"]:.2f}  cheap B=2 {N * cr + 2 * cu:.2f}  B=16 {N * cr + 16 * cu:.2f}  '
      f'speedup {cost["full_uni"] / (N * cr + 16 * cu):.1f}-{cost["full_uni"] / (N * cr + 2 * cu):.1f}x')

fig, a = plt.subplots(figsize=(3.4, 3.5))
a.errorbar(cost['full_uni'], m('full_uni'), sd('full_uni'), color='k', marker='*', ms=9, capsize=2, ls='none', label='full-slide MIL, UNI')
a.scatter([], [], color='C7', marker='*', s=30, label='full-slide MIL, ResNet50 (per seed)')
series = [('teach_uni', lambda K: N * cu, 'UNI students $\\to$ UNI teacher', 'C1', 's'),
          ('teach_r50', lambda K: N * cr, 'ResNet50 students $\\to$ ResNet50 teacher', 'C2', 'v'),
          ('cheap2uni', lambda K: N * cr + K * cu, 'ResNet50 students $\\to$ UNI teacher', 'C3', 'D')]
for p, cf, lab, col, mk in series:
    xs = np.array([cf(K) for K in KS]) * (1 + 0.06 * (np.arange(4) - 1.5) if p in ('teach_uni', 'teach_r50') else 1)
    a.errorbar(xs, [m(f'{p}_K{K}') for K in KS], [sd(f'{p}_K{K}') for K in KS], color=col, marker=mk, ms=4, capsize=1.5, lw=.8, label=lab)

# sparse-reading students (exp_sparse.py log): R50 students encode a fraction of the slide (stride-s grid, +adaptive zoom), UNI teacher
import re, ast
SP = {}
for l in open(f'{D}/sparse_log.txt'):
    mm = re.match(r'K(\d+) \d+s (\{.*\})', l)
    if mm:
        for k, v in ast.literal_eval(mm.group(2)).items(): SP.setdefault(k, []).append(v)
SK = int(os.environ.get('SPARSE_K', 8))
for p, col, mk, lab in [('adapt', 'C3', 'P', 'sparse-reading students (+zoom) $\\to$ UNI'), ('grid', 'C5', 'x', 'sparse-reading students (grid) $\\to$ UNI')]:
    pts = sorted([(np.mean([v[1] for v in SP[f'{p}{s}_K{SK}']]), np.mean([v[0] for v in SP[f'{p}{s}_K{SK}']]), np.std([v[0] for v in SP[f'{p}{s}_K{SK}']]))
                  for s in (2, 3, 4) if f'{p}{s}_K{SK}' in SP])
    xs = [N * fr_ * cr + SK * cu for fr_, _, _ in pts]
    a.errorbar(xs, [q[1] for q in pts], [q[2] for q in pts], color=col, marker=mk, ms=5, capsize=1.5, lw=.8, ls='--', label=lab + f' ($B$={SK})')

# compute-matched baselines without students (exp_cascade.py)
C = [json.load(open(f'{D}/cascade_s{s_}.json'))['slide'] for s_ in range(6)]
cm = lambda k: np.mean([c[k]['balacc'] for c in C]); cs = lambda k: np.std([c[k]['balacc'] for c in C])
a.errorbar([N / s_ ** 2 * cu for s_ in (4, 3, 2)], [cm(f'uni_grid{s_}') for s_ in (4, 3, 2)], [cs(f'uni_grid{s_}') for s_ in (4, 3, 2)],
           color='k', marker='o', mfc='w', ms=4, capsize=1.5, lw=.8, ls=':', label='UNI MIL on 1/16, 1/9, 1/4 grid')
xc = N * cr * 1.25
a.scatter([xc] * 6, [c['cascade_K8']['balacc'] for c in C], color='C4', marker='^', s=18, zorder=3, label='cascade: R50 MIL top-8 $\\to$ UNI (per seed)')
a.scatter([xc] * 6, [c['cascade_attacked_K8']['balacc'] for c in C], facecolor='w', edgecolor='C4', marker='^', s=18, zorder=3, label='cascade, compromised selector (per seed)')
a.scatter([N * cr * .8] * 6, [c['full_r50']['balacc'] for c in C], color='C7', marker='*', s=30, zorder=3)
# published MIL baselines (torchmil, tab_published.py): per-method mean over 3 seeds, full-slide reading
import sys; sys.path.insert(0, 'analysis'); from tab_published import BASE
MK = {'abmil': 'o', 'clam_sb': 's', 'dsmil': 'v', 'transmil': 'D', 'dtfdmil': 'h'}
for enc, x0, col, lab in [('uni', N * cu, 'C0', 'published MIL, full UNI (5 methods)'), ('r50bt', N * cr, 'C9', 'published MIL, full ResNet50 (5 methods)')]:
    ms = [m_ for m_ in MK if (enc, m_) in BASE]
    for j_, m_ in enumerate(ms):
        a.scatter(x0 * (1 + .07 * (j_ - (len(ms) - 1) / 2)), BASE[(enc, m_)]['balacc'].mean(), facecolor='none', edgecolor=col, marker=MK[m_], s=16, lw=.8, zorder=2,
                  label=lab if j_ == 0 else None)
a.set_xscale('log'); a.set_xlabel(unit, fontsize=8); a.set_ylabel('balanced accuracy', fontsize=8); a.set_ylim(.4, 1.0); a.set_xlim(.08, 15)
a.tick_params(labelsize=7); a.grid(alpha=.3); a.legend(fontsize=5.5, loc='upper center', bbox_to_anchor=(.45, -.18), ncol=2, frameon=False)
plt.tight_layout(); plt.savefig('paper_v1/figs/fig_cheap.pdf', bbox_inches='tight')

# LaTeX table rows
f = lambda k: f'{m(k):.3f}{{\\tiny$\\pm${sd(k):.3f}}}'.replace('0.', '.')
print('B & random$\\to$UNI & UNI$\\to$UNI & R50$\\to$R50 & R50$\\to$UNI & R50$\\to$UNI (look-ahead DA) & blank')
for K in KS:
    print(f'{K} & ' + ' & '.join(f(f'{p}_K{K}') for p in ('random_uni', 'teach_uni', 'teach_r50', 'cheap2uni', 'cheap2uni_liar_rollout', 'cheap2uni_blank')) + ' \\\\')
print('full UNI', f('full_uni'), 'full R50', f('full_r50'), [round(j['slide']['full_r50']['balacc'], 2) for j in J])
print('cost per slide: full UNI %.3g, R50 students + B UNI: ' % cost['full_uni'] + ', '.join(f'B={K}: {N * cr + K * cu:.3g}' for K in KS))
