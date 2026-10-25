"""Cross-domain robustness figure: one compromised evidence-selection component, three natural-image and three pathology settings.
Bars per setting: single selector clean / compromised; advocates with the true advocate compromised; hybrid clean / worst single compromise.
VLM (letter scoring, results/vlm/*_deb2/_deb3[ad].jsonl): hybrid worst = per question, correct only if correct under every compromised slot
  (adaptive judge-aware attacker if *_deb3ad.jsonl exists, else the fixed attacker tile). Pathology (results/ext/*_ext_atty.json, 3 seeds x 5 folds):
  attacked = worst of per-tile and adaptive set attacks over all slots incl. the true class's advocate.
usage: python analysis/fig_robust_all.py -> paper/figs/fig_robust_all.{pdf,png} + results/robust_all.json"""
import os, json, glob, re, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); O = f'{B}/paper/figs'; os.makedirs(O, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

def vlm(tag):
    ld = lambda f: {r['qid']: r for r in map(json.loads, open(f))}
    D2 = ld(f'{B}/results/vlm/{tag}_deb2.jsonl'); D3 = ld(f'{B}/results/vlm/{tag}_deb3.jsonl')
    fa = f'{B}/results/vlm/{tag}_deb3ad.jsonl'; DA = ld(fa) if os.path.exists(fa) else D3
    fs = f'{B}/results/vlm/{tag}_singlead.jsonl'; DS = ld(fs) if os.path.exists(fs) else None  # judge-aware attack on the single selector
    q = [k for k in D3 if k in D2 and k in DA and (DS is None or k in DS)]; ok = lambda D, i, k: np.argmax(D[i][k]) == D[i]['y']
    m = lambda f: float(np.mean([f(i) for i in q]))
    return dict(single=m(lambda i: ok(D2, i, 'coopn')), single_att=m(lambda i: ok(DS, i, 'attn_ad') if DS else ok(D2, i, 'attn')), adv_att=m(lambda i: ok(DA, i, 'nway_cy')),
                hyb=m(lambda i: ok(D3, i, 'hyb')), hyb_att=m(lambda i: ok(DA, i, 'hyb_cy') and ok(DA, i, 'hyb_cn')), n=len(q), adaptive=DA is not D3, single_adaptive=DS is not None)

def wsi(ds, met):
    V = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f'{B}/results/ext/{ds}_*_ext_atty.json'):
        s = int(re.search(r'_s(\d+)_ext', f).group(1)); R = json.load(open(f))['res']
        for k, v in R.items():
            mm = re.match(r'x_hyb(\d+)(_\w+)?$', k)
            if mm and mm.group(1) != '4': k = 'hybK' + (mm.group(2) or '')
            V[k][s].append(v[met])
    m = lambda k: float(np.mean([np.mean(v) for v in V[k].values()]))
    return dict(single=m('admil'), single_att=m('admil_wb_worst'), adv_att=m('x_base_ad'), hyb=m('hybK'), hyb_att=m('hybK_ad'), n=len(V['admil']))

SET = [('V* · Qwen2.5-VL', vlm('vstar_q25')), ('V* · Qwen3-VL', vlm('vstar_q3')), ('HR-4K · Qwen2.5-VL', vlm('hr4k_q25')),
       ('CAMELYON17 (κ)', wsi('c17', 'kappa')), ('TCGA-RCC (BA)', wsi('tcga_rcc', 'balacc')), ('TCGA-NSCLC (BA)', wsi('tcga_nsclc', 'balacc'))]
json.dump({n: d for n, d in SET}, open(f'{B}/results/robust_all.json', 'w'), indent=1)
BARS = [('single', 'single selector', '#4C72B0', ''), ('single_att', 'single selector, compromised', '#4C72B0', '////'),
        ('adv_att', 'advocates, true advocate compromised', '#8172B2', '////'), ('hyb', 'hybrid (ours)', '#C44E52', ''),
        ('hyb_att', 'hybrid, worst single compromise', '#C44E52', '////')]
fig, a = plt.subplots(figsize=(7.0, 2.0)); w = .16
for j, (name, d) in enumerate(SET):
    for k, (key, lab, c, h) in enumerate(BARS):
        a.bar(j + (k - 2) * w, d[key], w, color=c if not h else 'white', edgecolor=c, hatch=h, lw=.6, label=lab if j == 0 else None)
a.axhline(0, color='k', lw=.4); a.axvline(2.5, color='#bbbbbb', lw=.6, ls='--')
a.set_xticks(range(len(SET))); a.set_xticklabels([n for n, _ in SET], fontsize=6); a.set_ylim(-.3, 1.0); a.set_ylabel('accuracy / score')
a.text(1, .97, 'natural images (VLM judge)', ha='center', fontsize=6, color='#555555'); a.text(4, .97, 'pathology (MIL judge)', ha='center', fontsize=6, color='#555555')
a.legend(fontsize=5.5, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.14))
plt.tight_layout(); plt.savefig(f'{O}/fig_robust_all.pdf', bbox_inches='tight'); plt.savefig(f'{O}/fig_robust_all.png', dpi=200, bbox_inches='tight')
for n, d in SET: print(f"{n:22s} " + '  '.join(f"{k}={d[k]:.3f}" for k, *_ in BARS) + f"  n={d['n']}" + (f"  adaptive={d.get('adaptive')}" if 'adaptive' in d else ''))
