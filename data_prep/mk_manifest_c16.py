"""Manifests for CAMELYON16 torchmil features (official split; test -> testA). One per encoder."""
import csv, os, numpy as np
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
D = ROOT + "/data/camelyon16/torchmil"; O = "/scratch/ciro/shared_baselines/manifests"; os.makedirs(O, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(D, 'splits.csv'))))
for enc, fd in (('uni', 'features'), ('r50bt', 'features_r50bt')):
    with open(os.path.join(O, f'c16_{enc}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['slide', 'path', 'label', 'split', 'dataset', 'center'])
        for r in rows:
            p = os.path.join(D, fd, r['bag_name'] + '.npy')
            if not os.path.exists(p): continue
            y = int(np.load(os.path.join(D, 'labels', r['bag_name'] + '.npy')).reshape(-1)[0])
            w.writerow([r['bag_name'], p, y, 'train' if r['split'] == 'train' else 'testA', 'camelyon16', ''])
    print(enc, sum(1 for _ in open(os.path.join(O, f'c16_{enc}.csv'))) - 1)
