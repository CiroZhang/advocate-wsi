"""Offline VLM analyses for the reviewer: transitions (c->w vs w->c), realized vs best-of natural zooms, bootstrap CIs, crop-necessary subset."""
import json, numpy as np, sys
D = "/Users/cirozhang/Desktop/idea/WSI_debate/narval_results/vlm_raw"
rs_ = lambda f: [json.loads(l) for l in open(f"{D}/{f}")]
am = lambda v: int(np.argmax(v)); B = 2000; R = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); bs = [x[R.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return f"{x.mean():.3f} [{np.quantile(bs, .025):.3f},{np.quantile(bs, .975):.3f}] n={len(x)}"
out = {}
for nm, f in (("V* Qwen2.5", "vstar_partial.jsonl"), ("V* Qwen3", "vstar_q3_partial.jsonl"), ("HR4K Qwen2.5", "hr4k_q25_partial.jsonl")):
    rs = rs_(f); o = {}
    g = np.array([am(r['base']['global_only']) == r['y'] for r in rs]); o['global_only'] = ci(g); o['full_hi'] = ci([am(r['base']['full_hi']) == r['y'] for r in rs])
    # single tile + global, relative to the global-only answer
    cw_any, wc_any, cw_rand, wc_rand = [], [], [], []
    for r, gc in zip(rs, g):
        pred = np.array(r['S_glob']).argmax(1); ok = pred == r['y']
        if gc: cw_any.append((~ok).any()); cw_rand.append((~ok).mean())
        else: wc_any.append(ok.any()); wc_rand.append(ok.mean())
    o['adv tile: some tile makes correct->wrong (among global-correct)'] = ci(cw_any)
    o['some tile makes wrong->correct (among global-wrong)'] = ci(wc_any)
    o['random tile: P(correct->wrong)'] = ci(cw_rand); o['random tile: P(wrong->correct)'] = ci(wc_rand)
    # realized deployed trajectory (self-zoom) vs best-of natural candidates
    sz = [(r, gc) for r, gc in zip(rs, g) if 'self' in r['zoom']]
    o['self-zoom realized: correct->wrong (among global-correct)'] = ci([am(r['zoom']['self'][1]) != r['y'] for r, gc in sz if gc])
    o['self-zoom realized: wrong->correct (among global-wrong)'] = ci([am(r['zoom']['self'][1]) == r['y'] for r, gc in sz if not gc])
    o['best-of natural candidates: some zoom correct->wrong (among global-correct)'] = ci([any(am(v[1]) != r['y'] for v in r['zoom'].values()) for r, gc in zip(rs, g) if gc and r['zoom']])
    o['oracle 1v1 debate (honest side = true answer), min order'] = ci([min(am(r['debate']['glob_honestfirst']) == r['y'], am(r['debate']['glob_liarfirst']) == r['y']) for r in rs])
    out[nm] = o
# DeepEyes transitions and crop-necessary subset (crop-necessary := Qwen2.5 global-only wrong AND GT tile + global right)
q25 = {r['qid']: r for r in rs_("vstar_partial.jsonl")}
need = {k for k, r in q25.items() if am(r['base']['global_only']) != r['y'] and np.array(r['S_glob'])[int(np.argmax(r['gt_overlap']))].argmax() == r['y']}
for nm, f in (("DeepEyes full-res", "deepeyes_v3_full_partial.jsonl"), ("DeepEyes 768 overview", "deepeyes_v3_lowres_partial.jsonl")):
    rs = rs_(f); o = {}
    nat = np.array([r['native']['pred'] == r['y'] for r in rs]); o['native'] = ci(nat)
    for s in ('gt', 'adv0', 'rand0'):
        sp = np.array([r['subst'][s]['pred'] == r['y'] for r in rs])
        o[f'{s}: acc'] = ci(sp); o[f'{s}: correct->wrong'] = ci(~sp[nat]); o[f'{s}: wrong->correct'] = ci(sp[~nat])
    sub = [r for r in rs if r['qid'] in need]
    o[f'crop-necessary subset (n={len(sub)}): native'] = ci([r['native']['pred'] == r['y'] for r in sub])
    o['crop-necessary: GT crop'] = ci([r['subst']['gt']['pred'] == r['y'] for r in sub])
    o['crop-necessary: any adversarial crop -> wrong'] = ci([any(r['subst'][f'adv{i}']['pred'] != r['y'] for i in range(4)) for r in sub])
    o['crop-necessary: random crop'] = ci([r['subst']['rand0']['pred'] == r['y'] for r in sub])
    out[nm] = o
for k, o in out.items():
    print('##', k); [print(f'  {a:80s} {b}') for a, b in o.items()]
json.dump(out, open("/Users/cirozhang/Desktop/idea/WSI_debate/analysis/vlm_analysis.json", "w"), indent=1)
