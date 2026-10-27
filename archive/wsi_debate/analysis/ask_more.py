"""Simulated 'attending asks for more evidence': start at the smallest budget; if the judge's |logit| < tau, escalate to the next budget.
Uses saved per-budget judge logits (each budget has its own on-policy judge), so this is an offline approximation of an adaptive protocol."""
import numpy as np, sys
KS = [2, 4, 8, 16]
def ba(y, p): return 0.5 * ((p[y == 1] > 0).mean() + (p[y == 0] <= 0).mean())
for pref in ['c16_logits', 'c16_warm_logits']:
    print('##', pref)
    for proto in ['debate', 'debate_liar_rollout', 'consult_liar', 'attn_top_attacked', 'random']:
        rows = []
        for tau in [0, 0.5, 1, 2, 3, 1e9]:
            accs, costs, fixed = [], [], []
            for s in range(3):
                d = np.load(f'narval_results/logits/{pref}_s{s}.npz'); y = d['y']
                L = np.stack([d[f'{proto}_K{k}'] for k in KS], 1)
                stop = np.array([next((i for i in range(len(KS)) if abs(L[j, i]) >= tau), len(KS) - 1) for j in range(len(y))])
                p = L[np.arange(len(y)), stop]; accs.append(ba(y, p)); costs.append(np.mean(np.array(KS)[stop]))
            rows.append(f'tau={tau:g}: BA {np.mean(accs):.3f}±{np.std(accs):.3f} avgK {np.mean(costs):.1f}')
        fx = ' | '.join(f'K{k} {np.mean([ba(np.load(f"narval_results/logits/{pref}_s{s}.npz")["y"], np.load(f"narval_results/logits/{pref}_s{s}.npz")[f"{proto}_K{k}"]) for s in range(3)]):.3f}' for k in KS)
        print(f'  {proto:22s} fixed: {fx}'); [print('     ', r) for r in rows]
