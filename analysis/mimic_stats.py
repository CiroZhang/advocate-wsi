"""How often does a slide contain enough real tiles that look like another class? Per test slide (5 folds, seed 0), count the tiles
whose per-tile class under the full-slide CONCH ABMIL (argmax of its per-tile class logits) differs from the slide label. Supports
Prop. 1(i): the single selector is robust only if no K tiles lead the judge to a wrong class.
usage: python analysis/mimic_stats.py SEL_DIR  (SEL_DIR holds tcga_{rcc,nsclc}_b0_k4_f*_s0_ext_red_sel.pt and tcga_*/manifest.tsv)
       -> results/ext/mimic_stats.json"""
import sys, os, glob, csv, json, numpy as np, torch
D = sys.argv[1]; B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); out = {}
for co, PR, K in [('rcc', ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH'], 12), ('nsclc', ['TCGA-LUAD', 'TCGA-LUSC'], 8)]:
    lab = {r['file_name'][:-4]: PR.index(r['project']) for r in csv.DictReader(open(f'{D}/tcga_{co}/manifest.tsv'), delimiter='\t')}
    n_oth, frac, y, pair = [], [], [], np.zeros((len(PR), len(PR)))
    for f in sorted(glob.glob(f'{D}/tcga_{co}_b0_k4_f*_s0_ext_red_sel.pt')):
        d = torch.load(f, weights_only=False)
        for nm, ts in zip(d['names'], d['tile_subtype']):
            yy = lab[nm]; c = np.bincount(ts.numpy(), minlength=len(PR)); n_oth.append(len(ts) - c[yy]); frac.append(1 - c[yy] / len(ts)); y.append(yy)
            pair[yy] += c >= K
    n_oth, frac, y = map(np.array, (n_oth, frac, y))
    out[co] = {'n_slides': int(len(y)), 'K': K, 'frac_slides_ge_K_other': float((n_oth >= K).mean()), 'frac_slides_ge_1_other': float((n_oth >= 1).mean()),
               'median_other_tiles': float(np.median(n_oth)), 'median_other_frac': float(np.median(frac)),
               'per_class_ge_K': {PR[a]: {PR[b]: float(pair[a, b] / max((y == a).sum(), 1)) for b in range(len(PR)) if b != a} for a in range(len(PR))}}
    print(co, json.dumps(out[co], indent=1))
json.dump(out, open(f'{B}/results/ext/mimic_stats.json', 'w'), indent=1)
