"""Fig: teacher with vs without follow-up questions, BA vs patches inspected, per honest-student error rate (CAMELYON16, 3 seeds)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = 'narval_results/teach'; E = [0.0, 0.25, 0.5, 0.75]; KS = [2, 4, 8, 16]
fig, ax = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
for r, (tag, nm) in enumerate([('', 'UNI'), ('_r50bt', 'ResNet50-BT')]):
    J = [json.load(open(f'{D}/teach{tag}_s{s}.json'))['res'] for s in range(3)]
    m = lambda k, f='balacc': np.mean([j[k][f] for j in J]); sd = lambda k: np.std([j[k]['balacc'] for j in J])
    for c, e in enumerate(E):
        a = ax[r, c]
        y = [m(f'A_K{K}_eps{e}') for K in KS]; s = [sd(f'A_K{K}_eps{e}') for K in KS]
        a.errorbar(KS, y, s, color='k', marker='o', ms=4, capsize=2, label='silent teacher (fixed $B$)')
        for p, col, mk, lab in [('learned_lam', 'C3', 's', 'learned questions'), ('verify_tau', 'C0', '^', 'context question (conf.)'),
                                ('trail_tau', 'C2', 'v', 'ask trailing (conf.)'), ('rand_tau', 'C7', 'x', 'random patches (conf.)')]:
            ks = [k for k in J[0] if k.startswith(p) and k.endswith(f'_eps{e}')]
            xs = [m(k, 'patches') for k in ks]; ys = [m(k) for k in ks]; o = np.argsort(xs)
            a.plot(np.array(xs)[o], np.array(ys)[o], color=col, marker=mk, ms=5, lw=1 if p == 'learned_lam' else 0.6, ls='-' if p == 'learned_lam' else ':', label=lab)
        a.set_xscale('log', base=2); a.set_xticks(KS); a.set_xticklabels(KS); a.set_ylim(0.3, 1.0); a.grid(alpha=.3)
        if r == 0: a.set_title(f'honest student errs {int(e * 100)}%')
        if c == 0: a.set_ylabel(f'{nm}\nbalanced accuracy')
        if r == 1: a.set_xlabel('patches the teacher inspects')
ax[0, 0].legend(fontsize=7, loc='lower right')
plt.tight_layout(); plt.savefig('paper_v1/figs/fig_questions.pdf', bbox_inches='tight'); print('ok')
