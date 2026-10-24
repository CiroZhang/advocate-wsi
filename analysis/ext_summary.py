"""3-seed summary of the EXT experiments (results/ext/*_ext.json): per seed, mean over 5 folds; report mean ± sd over seeds.
C17 uses kappa, TCGA cohorts use balanced accuracy. 'hybK' = hybrid with K tiles (K = the full-slide-free budget, 12 or 16).
usage: python analysis/ext_summary.py [results/ext]"""
import sys, json, glob, re, collections, numpy as np

ROWS = [('admil', 'single selector, clean'), ('admil_wb_worst', 'single selector, attacked'), ('admil_weak', 'single selector, weak scorer'),
        ('teach', 'advocates, clean'), ('x_base_wb', 'advocates, per-tile attack'), ('x_base_ad', 'advocates, adaptive attack'),
        ('x_hyb4', 'hybrid (4 tiles), clean'), ('x_hyb4_ad', 'hybrid (4 tiles), adaptive attack'),
        ('hybK', 'hybrid (K tiles), clean'), ('hybK_wb', 'hybrid (K tiles), per-tile attack'), ('hybK_ad', 'hybrid (K tiles), adaptive attack'),
        ('x_indep', 'independent scorers, clean'), ('x_indep_wb', 'independent scorers, attacked'),
        ('x_weak', 'advocates, weak scorer'), ('x_weak_wb', 'advocates, weak scorer, attacked')]
DS = [('c17', 'kappa', 'C17 κ'), ('tcga_rcc', 'balacc', 'RCC BA'), ('tcga_nsclc', 'balacc', 'NSCLC BA')]

def main(d='results/ext'):
    V = collections.defaultdict(lambda: collections.defaultdict(list))  # (ds, key) -> seed -> fold values
    for f in glob.glob(f'{d}/*_ext.json'):
        b = f.split('/')[-1]; ds = next(x for x, _, _ in DS if b.startswith(x + '_')); met = dict((x, m) for x, m, _ in DS)[ds]
        seed = int(re.search(r'_s(\d+)_ext', b).group(1)); R = json.load(open(f))['res']
        for k, v in R.items():
            m = re.match(r'x_hyb(\d+)(_\w+)?$', k)
            if m and m.group(1) != '4': k = 'hybK' + (m.group(2) or '')
            V[(ds, k)][seed].append(v[met])
    seeds = sorted({s for v in V.values() for s in v})
    out = [f'# EXT experiments, seeds {seeds} (5 folds each; mean over folds, then mean ± sd over seeds)', '',
           '| | ' + ' | '.join(n for _, _, n in DS) + ' |', '|---' * (len(DS) + 1) + '|']
    for k, name in ROWS:
        cells = []
        for ds, _, _ in DS:
            per = [np.mean(v) for s, v in sorted(V[(ds, k)].items()) if len(v) == 5]
            cells.append(f'{np.mean(per):.3f} ± {np.std(per, ddof=1) if len(per) > 1 else 0:.3f} (n={len(per)})' if per else '–')
        out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    print('\n'.join(out)); return out

if __name__ == '__main__':
    main(*sys.argv[1:])
