"""C17 case finding: where do the level students beat an ADMIL-style single selector?
Per-stage recall (clean + attacked) and per-slide consistent wins across seeds.
usage: python c17_cases.py DIR   (DIR holds c17_b0_k4_f*_s*.npz)"""
import sys, glob, numpy as np
from collections import defaultdict
D = sys.argv[1]; ST = ['negative', 'itc', 'micro', 'macro']
KS = ['full_conch', 'teach', 'noflag', 'admil', 'teach_wb_worst', 'noflag_wb_worst', 'admil_wb_worst']
hit = defaultdict(lambda: defaultdict(list)); lab = {}
for f in sorted(glob.glob(f'{D}/c17_b0_k4_f*_s*.npz')):
    d = np.load(f)
    for k in KS:
        p = d[k].argmax(1)
        for n, yt, pt in zip(d['names'], d['y'], p): hit[k][str(n)].append(int(pt == yt)); lab[str(n)] = int(yt)
names = sorted(lab); y = np.array([lab[n] for n in names])
print('slides per stage', np.bincount(y, minlength=4))
print('\nper-stage recall (mean over seeds)')
for k in KS:
    r = np.array([np.mean(hit[k][n]) for n in names])
    print(f'{k:16s}', '  '.join(f'{s}:{r[y == c].mean():.2f}' for c, s in enumerate(ST)))
def wins(a, b, tag):
    ra = np.array([np.mean(hit[a][n]) for n in names]); rb = np.array([np.mean(hit[b][n]) for n in names])
    w = (ra >= 2 / 3) & (rb <= 1 / 3); l = (rb >= 2 / 3) & (ra <= 1 / 3)
    print(f'\n{tag}: {a} right & {b} wrong in >=2/3 seeds: {w.sum()} slides; reverse: {l.sum()}')
    for c, s in enumerate(ST): print(f'  {s:9s} wins {int((w & (y == c)).sum()):3d}  losses {int((l & (y == c)).sum()):3d}')
    return [n for n in np.array(names)[w]]
W_clean = wins('teach', 'admil', 'clean'); W_att = wins('teach_wb_worst', 'admil_wb_worst', 'attacked')
wins('teach', 'full_conch', 'vs full'); wins('teach', 'noflag', 'flags'); wins('teach_wb_worst', 'noflag_wb_worst', 'flags attacked')
print('\nclean wins over admil (slide, stage):'); [print(' ', n, ST[lab[n]]) for n in W_clean]
