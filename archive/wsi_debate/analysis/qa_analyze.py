"""Summaries for the WSI-QA teaching-round pilot.
usage: python qa_analyze.py <jsonl> [<jsonl> ...]  (SlideChat perturbation file or qa_teach output; detected by keys)"""
import sys, json, collections, numpy as np
for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path)]
    if not R: continue
    keys = [k for k in ('orig', 'shuf', 'para', 'noise', 'none', 'qonly', 'rand', 'qret', 'teach', 'teach_ul', 'att', 'blank', 'swap', 'teach_shuf', 'teach_para', 'qonly_shuf', 'qonly_para', 'verify') if k in R[0]]
    print(f'== {path}  n={len(R)}')
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        for t in (r['task'], 'ALL'):
            for k in keys: by[t][k].append(int(np.argmax(r[k]) == r['y']))
            ref = 'orig' if 'orig' in r else 'teach'
            for k in keys:
                if k != ref: by[t]['flip_' + k].append(int(np.argmax(r[k]) != np.argmax(r[ref])))
            by[t]['major'].append(r['y'])
    print('task'.ljust(22) + ''.join(k[:8].rjust(9) for k in keys) + '   | flips vs ' + ('orig' if 'orig' in keys else 'teach'))
    for t in sorted(by, key=lambda t: (t == 'ALL', t)):
        d = by[t]; row = t[:21].ljust(22) + ''.join(f'{np.mean(d[k]):9.3f}' for k in keys)
        row += '   | ' + ' '.join(f"{k[5:]}={np.mean(d[k]):.2f}" for k in d if k.startswith('flip_'))
        print(row, f'(n={len(d[keys[0]])})')
    for a, b in (('teach', 'teach_shuf'), ('teach', 'teach_para'), ('qonly', 'qonly_shuf'), ('qonly', 'qonly_para'), ('orig', 'shuf'), ('orig', 'para')):
        if a in R[0] and b in R[0]: print(f'flip {b} vs {a}: %.3f' % np.mean([np.argmax(r[a]) != np.argmax(r[b]) for r in R]))
    if 'sec' in R[0]: print('slidechat sec/question (one full-slide forward): %.3f' % np.mean([r['sec'] for r in R]))
    if 't_teacher' in R[0]:
        print('student sec %.4f  teacher sec %.3f  tiles/slide %.0f' % (np.mean([r['t_student'] for r in R]), np.mean([r['t_teacher'] for r in R]), np.mean([r['N'] for r in R])))
