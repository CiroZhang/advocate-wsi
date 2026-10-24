"""Summarize crop-substitution audits (agent_subst.py / de_attack.py outputs).
Per file: native accuracy; accuracy when every zoom returns the GT tile / random tiles / adversarial tiles / a blank image;
crop reliance = acc(GT) - acc(random); share of answers that change vs native; adversarial flips among natively-correct;
first-turn commitment (an answer is already stated before any zoom); 'confirmation' language when the crop shows nothing relevant.
usage: python agent_analyze.py FILE.jsonl [...]"""
import sys, json, re, glob, numpy as np
CONF = re.compile(r'(zoom|crop|closer|cropped)[^.]{0,80}(confirm|clearly|shows|reveal|indeed|visible)|confirm|indeed', re.I)
def summarize(path):
    R = [json.loads(l) for f in sorted(glob.glob(path)) for l in open(f)]
    if not R: return None
    acc = lambda k: np.mean([r['subst'][k]['pred'] == r['y'] for r in R if k in r['subst']])
    has = lambda k: all(k in r['subst'] for r in R)
    nat = np.mean([r['native']['pred'] == r['y'] for r in R]); rnd = np.mean([acc(k) for k in ('rand0', 'rand1') if has(k)])
    corr = [r for r in R if r['native']['pred'] == r['y']]
    out = dict(n=len(R), native=nat, tool_use=np.mean([r['native']['n_tool'] > 0 for r in R]), gt=acc('gt'), rand=rnd, reliance=acc('gt') - rnd,
               adv_any_flip=np.mean([any(r['subst'][f'adv{j}']['pred'] != r['y'] for j in range(4) if f'adv{j}' in r['subst']) for r in corr]),
               changed=np.mean([np.mean([v['pred'] != r['native']['pred'] for v in r['subst'].values()]) for r in R]))
    if has('blank'): out['blank'] = acc('blank')
    if 'pred_first' in R[0]['native']:
        out['first_turn_answer'] = np.mean([r['native']['pred_first'] is not None for r in R])
    off = [r['subst'][k]['text'] for r in R for k in ('rand0', 'rand1') if r['subst'][k].get('tile_gt_overlap', 1) == 0]
    out['confirm_on_irrelevant_crop'] = np.mean([bool(CONF.search(t)) for t in off]) if off else float('nan')
    if has('blank'): out['confirm_on_blank'] = np.mean([bool(CONF.search(r['subst']['blank']['text'])) for r in R])
    return out
if __name__ == '__main__':
    for p in sys.argv[1:]:
        s = summarize(p)
        if s: print(p, '\n  ' + '  '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in s.items()))
