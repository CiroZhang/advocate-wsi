import json, numpy as np
d = [json.load(open(f'narval_results/r50bt/c16_r50bt_warm_s{s}.json'))['slide'] for s in range(3)]
def m(k): v = [x[k]['balacc'] for x in d]; return f"{np.mean(v):.3f}±{np.std(v):.3f}  ({' '.join(f'{a:.2f}' for a in v)})"
print('full', m('full'), 'auc', [round(x['full']['auc'], 3) for x in d])
for B in (2, 4, 8, 16):
    print(f'B={B}', ' | '.join(f"{n}: {m(f'{k}_K{B}')}" for n, k in [('attn', 'attn_top'), ('attn att', 'attn_top_attacked'), ('cons adv', 'consult_liar'), ('randmask32', 'randmask_ens32'), ('debate', 'debate'), ('debate lookahead', 'debate_liar_rollout')]))
