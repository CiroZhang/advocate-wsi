import json, sys, numpy as np, collections as C
rs = [json.loads(l) for l in open("/scratch/ciro/cvpr_lab/paperA_vstar/vstar_partial.jsonl")]
am = lambda v: int(np.argmax(v))
out = C.defaultdict(list)
for r in rs:
    y = r['y']; A = np.array(r['S_alone']); G = np.array(r['S_glob']); ov = np.array(r['gt_overlap'])
    out['full_hi'].append(am(r['base']['full_hi']) == y); out['global_only'].append(am(r['base']['global_only']) == y)
    for nm, S in (('alone', A), ('glob', G)):
        pred = S.argmax(1); gti = int(ov.argmax())
        out[f'{nm}: GT tile'].append(pred[gti] == y)
        out[f'{nm}: random tile (mean)'].append((pred == y).mean())
        out[f'{nm}: attack, any tile -> wrong'].append((pred != y).any())
        hide = ov < 0.1
        out[f'{nm}: hide attack (tiles w/o target) -> wrong'].append((pred[hide] != y).any() if hide.any() else False)
        # targeted: can each wrong option be reached?
        wrong = [k for k in range(r['nopt']) if k != y]
        out[f'{nm}: targeted attack success (avg over wrong opts)'].append(np.mean([(pred == k).any() for k in wrong]))
        out[f'{nm}: frac tiles answering wrong'].append((pred != y).mean())
    z = r['zoom']
    if 'self' in z:
        out['self-zoom alone'].append(am(z['self'][0]) == y); out['self-zoom +global'].append(am(z['self'][1]) == y); out['self-zoom hits target'].append(z['self'][2] > 0.5)
    for k, v in z.items():
        if k.startswith('confirm_'):
            kk = int(k.split('_')[1])
            out['confirm-zoom alone: correct'].append(am(v[0]) == r['y']); out['confirm-zoom +global: correct'].append(am(v[1]) == r['y'])
            out['confirm-zoom alone: answers the confirmed (wrong) option'].append(am(v[0]) == kk); out['confirm-zoom +global: answers the confirmed option'].append(am(v[1]) == kk)
            out['confirm-zoom hits target'].append(v[2] > 0.5)
    # natural reach: any natural zoom (self or confirm) -> wrong answer, with global view
    nat = [am(v[1]) != y for v in z.values()]
    out['natural reach (+global): some natural zoom -> wrong'].append(any(nat) if nat else False)
    for k, v in r['debate'].items(): out[f'debate {k}'].append(am(v) == y)
print(f"n={len(rs)}  by category: {dict(C.Counter(r['cat'] for r in rs))}")
for k, v in out.items(): print(f"{k:60s} {np.mean(v):.3f}  (n={len(v)})")
